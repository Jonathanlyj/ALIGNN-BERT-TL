
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import glob
import os
import pandas as pd
from collections import defaultdict
import configparser


SEED = 1
# props = ['ehull', 'formation_energy_peratom', 'mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon']
# props = ['ehull','mbj_bandgap', 'slme', 'spillage', 'magmom_outcar', 'Tc_supercon']
# props = ['ehull', 'formation_energy_peratom']
props = ['formation_energy_peratom']


parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--input_dir', help='input data directory', default="./embeddings", type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo', 'combo'], default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--raw', action='store_true')
parser.add_argument('--save_data', action='store_true')
parser.add_argument('--gnn_only', action='store_true')
parser.add_argument('--gnn_file_path', type=str, required=False)
parser.add_argument('--split_dir', type=str, required=False)
parser.add_argument('--intersec_file', type=str, required=False)
args =  parser.parse_args()
config = configparser.ConfigParser()
config.read('config.ini')

def in_range(val, prop):
    upper = float(config[f'prop:{prop}']['upper'])
    lower = float(config[f'prop:{prop}']['lower'])
    return lower <= val <=upper

def prepare_dataset(args, prop):
    embeddings = []
    labels = []
    file_path = f"embeddings_{args.llm.replace('/', '_')}_{args.text}_*.csv"
    print(file_path)
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
    
    embed_file = glob.glob(file_path)
    if len(embed_file)>1:
        latest_file = max(embed_file, key=os.path.getctime)
        print("Latest file:", latest_file)
        embed_file = [latest_file]
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0)
    dat = data('dft_3d')
    ids = []
    if args.intersec_file:
        df_ids_interec = pd.read_csv(args.intersec_file)
        print(df_ids_interec.head())
        intersec_ids = df_ids_interec['jid'].values
    for i in tqdm(dat, desc="Preparing data"):
        if i[prop]!='na':
            if args.intersec_file:
                if i['jid'] not in intersec_ids:
                    continue
                
            if in_range(i[prop], prop) or args.raw:
                if i['jid'] in df_embed.index:
                    embeddings.append(df_embed.loc[i['jid']].values)
                    labels.append(i[prop])
                    ids.append(i['jid'])
    
    if args.save_data:
        num_cols = len(embeddings[0])
        col_names = [i for i in range(num_cols)]
        df_data = pd.DataFrame(embeddings, columns=col_names)
        df_data[prop] = labels
        df_data["ids"] = ids
        dataset_filename = f"dataset_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}"
        dataset_path = f"./data/{dataset_filename}"
        df_data['ids'] = df_data['ids'] + '.vasp'
        if args.gnn_file_path:
            df_gnn = pd.read_csv(args.gnn_file_path)
            dataset_path = f"./data/dataset_alignn_{args.llm.replace('/', '_')}_{args.text}_prop_{prop}"
            if args.gnn_only:
                df_gnn = pd.read_csv(args.gnn_file_path)
                df_gnn['id'] = df_gnn['id'] + '.vasp'
                dataset_path = f"./data/alignn_prop_{prop}"
                df_data = df_data[[prop, "ids"]].merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))
            else:
                df_gnn['id'] = df_gnn['id'] + '.vasp'
                df_data = df_data.merge(df_gnn, how='inner', left_on="ids", right_on="id", suffixes=('_lm', '_gnn'))
                print(df_data.head())
            df_data[prop] = df_data.pop(prop)
            df_data["ids"] = df_data.pop("ids")

        if args.split_dir:
            split_path = os.path.join(args.split_dir, f"dataset_split_{prop}.json")
            assert prop in split_path
            # for subset in ["test", "val", "train"]:
            #     sub_filename = f"{dataset_filename}_{subset}.csv"
            #     df_sub = pd.read_csv(os.path.join(args.split_dir, sub_filename))
            #     df_sub['ids'] = df_sub['ids'] + '.vasp'
            #     df_datasub = df_data[df_data["ids"].isin(df_sub["ids"])].drop(columns={"jid", "jid.1"})
            #     df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
            #     logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
            with open(split_path, 'r') as json_file:
                split_dic = json.load(json_file)
            for subset in ["test", "val", "train"]:
                sub_ids = [val+'.vasp' for val in split_dic[f"id_{subset}"]]
                
                df_datasub = df_data[df_data['ids'].isin(sub_ids)].drop(columns={"id", "full"}, errors='ignore')
                df_datasub.to_csv(f"{dataset_path}_{subset}.csv")
                print(f"{dataset_path}_{subset}: {len(df_datasub)}")
                logging.info(f"Saved {subset} dataset to {dataset_path}_{subset}.csv")
        
        else:
            logging.info(f"Constructed {df_data.shape[0]} samples for {prop} property")
            df_data.to_csv(f"{dataset_path}.csv")
            logging.info(f"Saved dataset to {dataset_path}")

    return embeddings, labels

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    for prop in props:
        prepare_dataset(args, prop)


