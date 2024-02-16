import os

import numpy as np
from pathlib import Path

import torch
from torch.nn import functional as F
from jarvis.db.figshare import data
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error
)
import pandas as pd
import logging
from datasets import Dataset
import argparse
import time
from collections import defaultdict
import configparser
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def in_range(val, prop):
    upper = float(config[f'prop:{prop}']['upper'])
    lower = float(config[f'prop:{prop}']['lower'])
    return lower <= val <=upper

def ensure_dir(dir_path):
    Path(dir_path).mkdir(parents=True, exist_ok=True)
    return dir_path
def get_datasets(test_file_path, args, prop, tokenizer_func, val_ratio = 0.1, test_ratio = 0.1):
    dat = data('dft_3d')
    df_prop = pd.DataFrame.from_dict(dat) 
    df_prop = df_prop[df_prop[prop] != 'na'][['jid',prop]]
    if not args.raw:
        df_prop = df_prop[df_prop[prop].apply(lambda x: in_range(x, prop))] 
    df_text = pd.read_csv(test_file_path, index_col = 0)
    df_samples = df_prop.merge(df_text, how = 'inner', on = 'jid')
    df_samples['label'] = df_samples[prop].astype(float)
    logging.info(f"Sample size for {prop}: {len(df_samples)}")
    train_dataset, test_dataset = train_test_split(df_samples, test_size = test_ratio, random_state = SEED)
    train_dataset, val_dataset = train_test_split(train_dataset, test_size = val_ratio / (1 - test_ratio), random_state = SEED)
    dataset_list = []
    for df_sample in [train_dataset, val_dataset, test_dataset]:
        dataset = Dataset.from_dict({
        'text': df_sample['text'],
        'label': df_sample['label']
        })
        tokenized_dataset = dataset.map(tokenizer_func, batched=True)
        dataset_list.append(tokenized_dataset)
    return dataset_list[0], dataset_list[1], dataset_list[2]

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

def get_config(model_name, model_revision="main"):
    # cache_dir = ensure_dir(cache_dir) if cache_dir else None
    config_kwargs = {
        "num_labels": 1,
        "revision": model_revision,
        "use_auth_token": None,
    }

    config = AutoConfig.from_pretrained(model_name, **config_kwargs)
    return config


def get_tokenizer(model_name, max_seq_length=1024, model_revision="main"):
    tokenizer_kwargs = {
        "use_fast": True,
        "revision": model_revision,
        "use_auth_token": None,
        "model_max_length": max_seq_length,
        "padding_side": 'right',  # Ensure this is a string key
    }

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)

    # Setting pad_token for GPT-2, as it does not have one by default
    if model_name == "gpt2" and tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    # For BERT and other models, the pad_token is usually set by default, so no need to change it

    return tokenizer



def get_optimizer(model, lr=3e-5, non_lm_lr=3e-4):
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not "bert" in n],
            "lr": non_lm_lr,
        },
        {"params": [p for n, p in model.named_parameters() if "bert" in n], "lr": lr},
    ]
    optimizer_kwargs = {
        "betas": (0.9, 0.999),
        "eps": 1e-8,
    }
    optimizer = AdamW(optimizer_grouped_parameters, **optimizer_kwargs)
    return optimizer

def get_trainer(
    train_dataset,
    val_dataset,
    tokenizer,
    model,
    optimizer,
    model_save_dir,
    num_epochs=10,
    lr=3e-5,
    metric_for_best_model="eval_loss",
):
    output_dir = ensure_dir(model_save_dir)

    training_args = TrainingArguments(
        num_train_epochs=num_epochs,
        output_dir=output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=4,# BERT: 16
        per_device_eval_batch_size=4,# BERT: 16,
        evaluation_strategy="epoch",
        load_best_model_at_end=False,
        save_total_limit=2,
        warmup_ratio=0.1,
        learning_rate=lr,
        seed=SEED,
        use_cpu=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),
    )
    return trainer


def get_predictions(output):
    preds, labels = output.predictions, output.label_ids
    return labels, preds.flatten()


parser = argparse.ArgumentParser(description='Fine tune llm on dataset')
parser.add_argument('--data_path', help='path to the dataset',default="./text/raw_0_75993.csv", type=str, required=True)
# parser.add_argument('--input_dir', help='input data directory', default=None, type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'], default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='bert-base-uncased', type=str,required=False)
# parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
# parser.add_argument('--label', help='target variable', default=None, type=str,required=False)
parser.add_argument('--raw', action='store_true')
args =  parser.parse_args()

config = configparser.ConfigParser()
config.read('config.ini')
DEVICE = torch.device("cuda")
# PROP = "mbj_bandgap"
# props = ['ehull',
#        'slme', 'spillage', 'magmom_outcar', "mbj_bandgap"]
# props = ['ehull','slme', 'spillage', 'magmom_outcar', "mbj_bandgap",
#          'formation_energy_peratom', 'Tc_supercon']
# props = ['ehull', 'magmom_outcar', "mbj_bandgap", 'formation_energy_peratom']
props = ['ehull', 'magmom_outcar', 'formation_energy_peratom']

# props = ['mbj_bandgap']
MODEL_NAME = args.llm 
TEXT = args.text
SEED = 0
LEARNING_RATE = 3e-5
NUM_EPOCHES = 10

def fine_tune(args):
    result = defaultdict(list)
    for PROP in props:
        start_time = time.time()
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark = False
        set_seed(SEED)

        dataset_path = args.data_path
        tokenizer = get_tokenizer(MODEL_NAME)
        def tokenize_func(examples):
            return tokenizer(examples["text"], padding="max_length", truncation=True)
        train_dataset, val_dataset, test_dataset = get_datasets(dataset_path, args, PROP, tokenize_func)
        config = get_config(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_config(config=config)
        if MODEL_NAME == 'gpt2':
            model.config.pad_token_id = model.config.eos_token_id
        model_save_dir = f"./models/fine_tune_{MODEL_NAME.replace('/','_')}_{PROP}_{TEXT}"

        optimizer = get_optimizer(model, lr=LEARNING_RATE)
        trainer = get_trainer(
            train_dataset,
            val_dataset,
            tokenizer,
            model,
            optimizer,
            model_save_dir,
            num_epochs=NUM_EPOCHES,
            lr=LEARNING_RATE,
        )

        train_result = trainer.train()
        logging.info(train_result)
        labels, predictions = get_predictions(trainer.predict(test_dataset))
        mse = mean_squared_error(labels, predictions)
        mae = mean_absolute_error(labels, predictions)
        logging.info(f"Prediction result for {PROP} MAE: {mae}")
        logging.info(f"Prediction result for {PROP} MSE: {mse}")
        logging.info(train_result)
        result['prop'].append(PROP)
        result['mae'].append(mae)
        result['mse'].append(mse)
        df_pred = pd.DataFrame({'labels': labels, 'predictions': predictions})
        df_pred.to_csv(f"./pred/ft_{args.llm}_{args.text}_{PROP}.csv")
        logging.info(f"Finished fune tuning {MODEL_NAME} {TEXT} for {PROP}")

        end_time = time.time()
        elapsed_time = end_time - start_time
        logging.info(f"Execution time: {elapsed_time} seconds ({elapsed_time/3600} hours)")
    df_rst = pd.DataFrame.from_dict(result)
    return df_rst

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    df_rst = fine_tune(args)
    filtered_str = '' if args.raw else '_filtered'
    output_csv = f"ft_{args.llm}_{args.text}_prop_{len(props)}_{filtered_str}.csv"
    df_rst.to_csv(output_csv)