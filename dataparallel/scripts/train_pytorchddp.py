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

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
os.environ['TOKENIZERS_PARALLELISM'] = "True"
        
def setup():
    # initialize the process group: 여러 노드에 있는 여러 프로세스가 동기화되고 통신합니다
    dist.init_process_group(backend="nccl")
    
    if 'WORLD_SIZE' in os.environ:
        # Environment variables set by torch.distributed.launch or torchrun
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
    elif 'OMPI_COMM_WORLD_SIZE' in os.environ:
        # Environment variables set by mpirun 
        world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        local_rank = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
    else:
        sys.exit("Can't find the evironment variables for local rank")

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
    return config

def cleanup():
    dist.destroy_process_group()

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
    device = torch.device("cuda", args.local_rank)
    
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

    # 미니배치가 겹치지 않게 함
    train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.rank)
    eval_sampler = DistributedSampler(eval_dataset, num_replicas=args.world_size, rank=args.rank)
     
    train_loader = DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, 
        num_workers=4*torch.cuda.device_count(), shuffle=False
    )    
    eval_loader = DataLoader(
        dataset=eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, 
        num_workers=4*torch.cuda.device_count(), shuffle=False
    )

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(device)
    model = DDP(model, device_ids=[device])
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_epochs * len(train_loader)
    args.num_training_steps = num_training_steps
    
    logging.info(f"num_training_steps: {num_training_steps}")
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )
    pbar = tqdm(total=num_training_steps, leave=True, desc="Training")    
    
    for epoch in range(1, args.num_epochs+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        
        train_model(args, device, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch, pbar)
        if args.rank == 0:
            eval_model(device, model, eval_loader)

    pbar.close()
    if args.model_dir and args.rank == 0:
        torch.save(model.cpu().state_dict(), os.path.join(args.model_dir, "model.pt"))
            
            
def train_model(args, device, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch, pbar):
    model.train()

    # AMP (Create gradient scaler)
    scaler = GradScaler(init_scale=16384)
    
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=args.use_fp16):
            outputs = model(**batch)
            loss = outputs.loss
                
        if args.use_fp16:
            # Backpropagation w/ gradient scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:                
            loss.backward()
            optimizer.step()
        
        lr_scheduler.step()
        pbar.update(1)
        
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            logging.info(f"[Epoch {epoch} {((epoch-1) * args.world_size)+batch_idx}/{args.num_training_steps}, Train loss: {loss.item()}")

def eval_model(device, model, eval_loader):
    model.eval()
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch['labels'].to(device)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
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
    
    start = time.time()
    main(args)     
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs)
    if config.rank == 0:
        logging.info(f"Elapsed time: {result}")
    cleanup()