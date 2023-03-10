import os
import sys
import torch
import transformers
import logging
import argparse
import time, datetime
from types import SimpleNamespace
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
import evaluate
from pprint import pformat
from tqdm.auto import tqdm, trange
from helper import TqdmLoggingHandler

os.environ['TOKENIZERS_PARALLELISM'] = "True"
        
def setup():
    
    world_size = 1
    rank = 0
    local_rank = 0
    device = torch.device("cuda", local_rank)    

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
        handlers=[TqdmLoggingHandler()])
    logging.info(f"Training begin. world_size: {world_size}")
        
    config = SimpleNamespace()
    config.world_size = world_size
    config.rank = rank
    config.local_rank = local_rank
    config.device = device    
    return config

def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--use_fp16", type=bool, default=True)    
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--model_id", type=str, default='bert-base-multilingual-cased')
    
    # SageMaker Container environment
    parser.add_argument("--train_dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--eval_dir", type=str, default=os.environ["SM_CHANNEL_EVAL"])
    parser.add_argument("--output_data_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
    parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument('--chkpt_dir', type=str, default='/opt/ml/checkpoints')     

    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args

def main(args):
    
    torch.manual_seed(args.seed)

#     train_dataset = load_dataset("nsmc", split="train")
#     eval_dataset = load_dataset("nsmc", split="test")
#     train_num_samples = 2000
#     eval_num_samples = 1000
#     train_dataset = train_dataset.shuffle(seed=42).select(range(train_num_samples))
#     eval_dataset = eval_dataset.shuffle(seed=42).select(range(eval_num_samples))

#     tokenizer = AutoTokenizer.from_pretrained(args.model_id)
#     def tokenize(batch):
#         return tokenizer(batch['document'], padding='max_length', max_length=128, truncation=True)

#     # tokenize dataset
#     train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['id', 'document'])
#     eval_dataset = eval_dataset.map(tokenize, batched=True, remove_columns=['id', 'document'])

#     # set format for pytorch
#     train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#     eval_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
#     train_dataset = train_dataset.rename_column("label", "labels")
#     eval_dataset = eval_dataset.rename_column("label", "labels")
    
    # load datasets
    train_dataset = load_from_disk(args.train_dir)
    eval_dataset = load_from_disk(args.eval_dir)

    logging.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logging.info(f" loaded test_dataset length is: {len(eval_dataset)}")
    logging.info(train_dataset[0])     

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.train_batch_size, 
        num_workers=0, shuffle=True
    )    
    eval_loader = DataLoader(
        dataset=eval_dataset, batch_size=args.eval_batch_size, 
        num_workers=0, shuffle=False
    )

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(args.device)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_epochs * len(train_loader)
    args.num_training_steps = num_training_steps
    
    logging.info(f"num_training_steps: {num_training_steps}")
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.num_epochs+1):
        train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch)
        if args.rank == 0:
            eval_model(args, model, eval_loader)

    if args.model_dir and args.rank == 0:
        torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pt"))
            
            
def train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch):
    model.train()
    epoch_pbar = tqdm(total=len(train_loader), colour="blue", leave=True, desc=f"Training epoch {epoch}") 
    
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(args.local_rank) for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        lr_scheduler.step()
        epoch_pbar.update(1)
        
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            logging.info(f"Train loss: {loss.item()}")

            
def eval_model(args, model, eval_loader):
    model.eval()
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    history = torch.zeros(3).to(args.local_rank)
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(args.local_rank) for k, v in batch.items()}
            labels = batch['labels'].to(args.local_rank)
            outputs = model(**batch)
            loss = outputs['loss']
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            history[0] += loss.item()
            history[1] += preds.eq(labels.view_as(preds)).sum().item()
            history[2] += len(batch['labels'])
            metrics.add_batch(predictions=preds, references=batch["labels"])

    logging.info(f"Eval. loss: {loss.item()}")
    logging.info(pformat(metrics.compute()))

if __name__ == "__main__":
    
    is_sm_container = True    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'train'
        eval_dir = 'eval'
        model_dir = 'model'
        output_data_dir = 'output_data'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_EVAL'] = f'{src_dir}/{eval_dir}'

    args = parser_args()
    config = setup() 
    args.world_size = config.world_size
    args.rank = config.rank
    args.local_rank = config.local_rank
    args.device = config.device
    
    start = time.time()
    main(args)     
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs)
    logging.info(f"Elapsed time: {result}")
