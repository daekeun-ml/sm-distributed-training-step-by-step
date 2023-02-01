import os
import torch
import transformers
import logging
import time, datetime
import argparse
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk, load_dataset
#import torch, torch_xla.core.xla_model as xm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
os.environ['TOKENIZERS_PARALLELISM'] = "True"
        
# compute metrics function for binary classification
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# Distributed training
def setup():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logging.info(f"Training begin")
       
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
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # load datasets
    train_dataset = load_from_disk(args.train_dir)
    eval_dataset = load_from_disk(args.eval_dir)

    logging.info(f" loaded train_dataset length is: {len(train_dataset)}")
    logging.info(f" loaded test_dataset length is: {len(eval_dataset)}")
    logging.info(train_dataset[0])
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_data_dir, exist_ok=True)    
    
    model = BertForSequenceClassification.from_pretrained(args.model_id, num_labels=2).to(device) 

    training_args = TrainingArguments(
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        warmup_steps=args.warmup_steps,
        fp16=args.use_fp16,
        metric_for_best_model="accuracy",
        output_dir=args.output_data_dir,
    )
    
    trainer = Trainer(
        model= model,
        args=training_args,
        train_dataset=train_dataset.with_format("torch"),
        eval_dataset=eval_dataset.with_format("torch"),
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    train_result = trainer.train()
    logging.info(train_result)

    eval_result = trainer.evaluate()
    logging.info(eval_result)

    # writes eval result to file which can be accessed later in s3 ouput
    with open(os.path.join(args.output_data_dir, "eval_results.txt"), "w") as writer:
        logging.info("***** Evaluation results *****")
        for key, value in sorted(eval_result.items()):
            writer.write(f"{key} = {value}\n")
            logging.info(f"{key} = {value}\n")

    # Saves the model to s3 uses os.environ["SM_MODEL_DIR"] to make sure checkpointing works
    trainer.save_model(args.model_dir)

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
    setup()
    start = time.time()
    main(args)  
    secs = time.time() - start
    result = datetime.timedelta(seconds=secs) 
    logging.info(f"Elapsed time: {result}")