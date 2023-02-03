import os
import sys
import torch
import transformers
import logging
import argparse
import functools
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
import torch.nn.functional as F

# FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,    
    transformer_auto_wrap_policy,
    enable_wrap,
    wrap,
)
        
def setup():

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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
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


def get_mp_policy(args):

    fp16_policy = MixedPrecision(
        param_dtype=torch.float16,
        # Gradient communication precision.
        reduce_dtype=torch.float16,
        # Buffer precision.
        buffer_dtype=torch.float16,
    )

    fp32_policy = None
    
    if args.use_fp16:
        return fp16_policy
    else:
        return fp32_policy
    

def cleanup():
    dist.destroy_process_group()

def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()

    # Default Setting
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--disable_tqdm", type=bool, default=True)
    parser.add_argument("--use_fp16", type=bool, default=False)    
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

    # 미니배치가 겹치지 않게 함
    train_sampler = DistributedSampler(train_dataset, rank=args.rank, num_replicas=args.world_size, shuffle=True)
    eval_sampler = DistributedSampler(eval_dataset, rank=args.rank, num_replicas=args.world_size)
     
    train_kwargs = {'batch_size': args.train_batch_size, 'sampler': train_sampler}
    eval_kwargs = {'batch_size': args.eval_batch_size, 'sampler': eval_sampler}
    cuda_kwargs = {'num_workers': 0, 'pin_memory': True, 'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    eval_kwargs.update(cuda_kwargs)
    
    train_loader = DataLoader(train_dataset, **train_kwargs)    
    eval_loader = DataLoader(eval_dataset, **eval_kwargs)
    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=100
    )
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(args.device)
    #model = DDP(model, device_ids=[args.local_rank])
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP
    model = FSDP(model, sharding_strategy=sharding_strategy, mixed_precision=get_mp_policy(args), device_id=torch.cuda.current_device())
    #, fsdp_auto_wrap_policy=my_auto_wrap_policy)
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_epochs * len(train_loader)
    args.num_training_steps = num_training_steps
    
    logging.info(f"num_training_steps: {num_training_steps}")
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    for epoch in range(1, args.num_epochs+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        
        train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch)
        eval_model(args, model, eval_loader)

    if args.model_dir:
        logging.info('==== Save Model ====')  
        dist.barrier()
        states = model.state_dict()
        if args.rank == 0:
            torch.save(states, os.path.join(args.model_dir, "model.pt"))
            
            
def train_model(args, model, train_loader, eval_loader, optimizer, lr_scheduler, epoch):
    model.train()
    ddp_loss = torch.zeros(2).to(args.rank)
    
    if args.rank == 0:
        epoch_pbar = tqdm(total=len(train_loader), colour="blue", leave=True, desc=f"Training epoch {epoch}")    
        
    for batch_idx, batch in enumerate(train_loader):
        batch = {k: v.to(args.local_rank) for k, v in batch.items()}
        optimizer.zero_grad()
        
        outputs = model(**batch)
        loss = outputs.loss
                  
        loss.backward()
        optimizer.step()

        ddp_loss[0] += loss.item()
        ddp_loss[1] += len(batch['labels'])
        
        lr_scheduler.step()
        if args.rank == 0:
            epoch_pbar.update(1)
        
        if batch_idx % args.log_interval == 0 and args.rank == 0:
            logging.info(f"Train loss: {loss.item()}")

    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    
    if args.rank == 0:
        avg_loss = ddp_loss[0]/ddp_loss[1]
        logging.info(f"Epoch: {epoch} End - \t Train Avg. Loss: {avg_loss:.6f}")
        epoch_pbar.close()
    
# I don't know why validation step for FSDP doesn't work. It freezes for tens of minutes and runs forever.
# https://github.com/facebookresearch/fairseq/issues/3532
# https://github.com/pytorch/pytorch/issues/82206
def eval_model(args, model, eval_loader):
    model.eval()
    
    correct = 0
    metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    ddp_loss = torch.zeros(3).to(args.rank)
    
    if args.rank == 0:
        epoch_pbar = tqdm(total=len(eval_loader), colour="green", leave=True, desc="Validation epoch")    
            
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.to(args.local_rank) for k, v in batch.items()}
            labels = batch['labels'].to(args.local_rank)
            outputs = model(**batch)
            
            #outputs= model(input_ids=batch["input_ids"],attention_mask=batch["attention_mask"],labels=batch["labels"])
            if args.rank == 0:
                epoch_pbar.update(1)
                
            logits = outputs.logits
            # preds = torch.argmax(logits, dim=-1)
            loss = outputs.loss
            ddp_loss[0] += loss.item()
            #ddp_loss[0] += F.nll_loss(outputs, labels, reduction='sum').item()  # sum up batch loss
            preds = logits.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            ddp_loss[1] += len(batch['labels'])
            ddp_loss[2] += preds.eq(labels.view_as(preds)).sum().item()
            # loss = outputs.loss
            # logits = outputs.logits
            # preds = torch.argmax(logits, dim=-1)
            metrics.add_batch(predictions=preds, references=batch["labels"])
            
    dist.all_reduce(ddp_loss, op=dist.ReduceOp.SUM)
    avg_loss = ddp_loss[0]/ddp_loss[1]
    
    if args.rank == 0:
        epoch_pbar.close()
        logging.info(f"Valid Avg. Loss: {avg_loss:.6f} \t Accuracy: ({int(ddp_loss[2])}/{int(ddp_loss[1])})")             
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
    if config.rank == 0:
        logging.info(f"Elapsed time: {result}")
    cleanup()