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
import re

# Distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
os.environ['TOKENIZERS_PARALLELISM'] = "True"

def setup(backend="nccl"):

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
        
    # initialize the process group: 여러 노드에 있는 여러 프로세스가 동기화되고 통신합니다
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)    
    device = torch.device("cuda", local_rank)    
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if rank == 0 else logging.WARNING,
        handlers=[TqdmLoggingHandler()])
    logging.info(f"Initialized the distributed environment. world_size={world_size}, rank={rank}, local_rank={local_rank}")
        
    config = SimpleNamespace()
    config.world_size = world_size
    config.rank = rank
    config.local_rank = local_rank
    config.device = device
    return config

def cleanup():
    dist.destroy_process_group()    

def _load_chkpt(args, model, optimizer):
    map_location = {'cuda:%d' % 0: 'cuda:%d' % args.local_rank}    
    chkpt_files = [file for file in os.listdir(args.chkpt_dir) if file.endswith('.pt')]
    epochs = [re.search('(\.*[0-9])(?=\.)',file).group() for file in chkpt_files]
    
    max_epoch = max(epochs)
    max_epoch_index = epochs.index(max_epoch)
    max_epoch_filename = chkpt_files[max_epoch_index]
    chkpt_path = os.path.join(args.chkpt_dir, max_epoch_filename)
    
    logging.info(f"Loading Checkpoint From: {chkpt_path}")
    chkpt = torch.load(chkpt_path, map_location=map_location)
    model.load_state_dict(chkpt["model"])
    optimizer.load_state_dict(chkpt["optimizer"])
    latest_epoch = chkpt["epoch"] + 1
    logging.info(f"Loaded checkpoint. Resuming training from epoch: {latest_epoch}")
    return model, optimizer, latest_epoch

def _save_chkpt(args, model, optimizer, epoch):
    
    if args.rank == 0:
        chkpt_path = os.path.join(args.chkpt_dir, f"chkpt-{epoch}.pt")
        logging.info("Saving the Checkpoint: {}".format(chkpt_path))
        state = {
            "epoch": epoch,
            "model": model.module.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(state, chkpt_path)
        
def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--num_epochs", type=int, default=3)
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
    parser.add_argument('--chkpt_dir', type=str, default=os.environ["SM_CHECKPOINTS"])    

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

    # 미니배치가 겹치지 않게 함
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)
     
    train_loader = DataLoader(
        dataset=train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, 
        num_workers=0, shuffle=False
    )    
    eval_loader = DataLoader(
        dataset=eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, 
        num_workers=0, shuffle=False
    )
        
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
        
    # Check if checkpoints exists
    if len(os.listdir(args.chkpt_dir)) > 0:
        model, optimizer, latest_epoch = _load_chkpt(args, model, optimizer)
    else:
        latest_epoch = 1
        
    model = DDP(
        model, 
        device_ids=[args.local_rank]
    )
    
    num_training_steps = (args.num_epochs - latest_epoch + 1) * len(train_loader)
    args.num_training_steps = num_training_steps
    logging.info(f"Number of Training steps: {num_training_steps}")    
    
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(latest_epoch, args.num_epochs+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch)
        eval_model(args, model, eval_loader)
        _save_chkpt(args, model, optimizer, epoch)        

    # Save model - only save on rank 0        
    if args.model_dir and args.rank == 0:
        model_filepath = os.path.join(args.model_dir, "multimodal_model.pt")
        logging.info('==== Save Model ====')        
        torch.save(model.state_dict(), model_filepath)
            
    dist.barrier()
    if args.rank == 0:
        logging.info("Distributed Training finished successfully!")        
            
def train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch):
    model.train()
    history = torch.zeros(2).to(args.local_rank)
    
    # AMP (Create gradient scaler)
    scaler = GradScaler(init_scale=16384)

    if args.rank == 0:
        epoch_pbar = tqdm(total=len(train_loader), colour="blue", leave=True, desc=f"Training epoch {epoch}")    
        
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(args.local_rank) for k, v in batch.items()}
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
        
        history[0] += loss.item()
        history[1] += len(batch['labels'])   
    
        if args.rank == 0:
            epoch_pbar.update(1)
        
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            logging.info(f"Train loss: {loss.item()}")
            
    dist.all_reduce(history, op=dist.ReduceOp.SUM)

    if args.rank == 0:
        avg_loss = history[0] / history[1]
        logging.info(f"Train avg. loss: {avg_loss:.6f}")            
        epoch_pbar.close()
        
        
def eval_model(args, model, eval_loader):
    model.eval()
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    history = torch.zeros(3).to(args.local_rank)

    if args.rank == 0:
        eval_pbar = tqdm(total=len(eval_loader), colour="green", leave=True, desc=f"Evaluation")    
    
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(args.local_rank) for k, v in batch.items()}
            labels = batch['labels'].to(args.local_rank)
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)
            history[0] += loss.item()
            history[1] += len(batch['labels'])   
            history[2] += preds.eq(labels.view_as(preds)).sum().item()
         
            metrics.add_batch(predictions=preds, references=batch["labels"])
            if args.rank == 0:
                eval_pbar.update(1)

    dist.all_reduce(history, op=dist.ReduceOp.SUM)

    if args.rank == 0:
        avg_loss = history[0] / history[1]
        logging.info(f"Eval avg. loss: {avg_loss:.6f}, # of correct: ({int(history[2])}/{int(history[1])})")          
        logging.info(pformat(metrics.compute()))    
        eval_pbar.close()    

if __name__ == "__main__":
    
    is_sm_container = True    
    if os.environ.get('SM_CURRENT_HOST') is None:
        is_sm_container = False        
        train_dir = 'train'
        eval_dir = 'eval'
        model_dir = 'model'
        output_data_dir = 'output_data'
        chkpt_dir = 'checkpoints'
        src_dir = '/'.join(os.getcwd().split('/')[:-1])
        #src_dir = os.getcwd()
        os.environ['SM_MODEL_DIR'] = f'{src_dir}/{model_dir}'
        os.environ['SM_OUTPUT_DATA_DIR'] = f'{src_dir}/{output_data_dir}'
        os.environ['SM_CHANNEL_TRAIN'] = f'{src_dir}/{train_dir}'
        os.environ['SM_CHANNEL_EVAL'] = f'{src_dir}/{eval_dir}'
        os.environ['SM_CHECKPOINTS'] = f'{src_dir}/{chkpt_dir}'        
    else:
        os.environ['SM_CHECKPOINTS'] = '/opt/ml/checkpoints'        

    args = parser_args()
    config = setup(backend="nccl") 
    args.world_size = config.world_size
    args.rank = config.rank
    args.local_rank = config.local_rank
    args.device = config.device
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True) 
    os.makedirs(args.chkpt_dir, exist_ok=True)         
        
    start = time.time()
    main(args)     
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs)
    if config.rank == 0:
        logging.info(f"Elapsed time: {result}")
    cleanup()