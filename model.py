#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
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
props = ['ehull','mbj_bandgap',
       'slme', 'spillage', 'magmom_outcar']

parser = argparse.ArgumentParser(description='run ml regressors on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--input_dir', help='input data directory', default="./embeddings", type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'], default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
parser.add_argument('--label', help='target variable', default=None, type=str,required=False)
parser.add_argument('--raw', action='store_true')

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
    file_path = f'embeddings_{args.llm}_{args.text}_*.csv'
    if args.input_dir:
        file_path = os.path.join(args.input_dir, file_path)
    embed_file = glob.glob(file_path)
    logging.info(f"Found embedding file: {embed_file}")
    df_embed = pd.read_csv(embed_file[0], index_col = 0)
    dat = data('dft_3d')
    for i in tqdm(dat, desc="Preparing data"):
        if i[prop]!='na':
            if in_range(i[prop], prop) or args.raw:
                if i['jid'] in df_embed.index:
                    embeddings.append(df_embed.loc[i['jid']].values)
                    labels.append(i[prop])
    logging.info(f"Constructed {len(labels)} samples for {prop} property")
    return embeddings, labels
   
# Main function
def run_regressor(args):
    global props
    if args.label:
        props = [args.label] #'formation_energy_peratom'#'exfoliation_energy'
    result = defaultdict(list)
    for prop in props:
        X, y = prepare_dataset(args, prop)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

        # Initialize and fit a linear regression model
        logging.info(f"Started fitting to model")
        regression_model = RandomForestRegressor(n_jobs = 32) #LinearRegression()
        regression_model.fit(X_train, y_train)

        # Predict using the test set
        y_pred = regression_model.predict(X_test)

        # Evaluate the model
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        logging.info(f"{prop}: mean_absolute_error: {mae}")
        # plt.plot(y_test, y_pred,'.')
        # plt.savefig('plot.png')
        # plt.close()
        logging.info(f"{prop}: Mean Squared Error: {mse}")
        result['prop'].append(prop)
        result['mae'].append(mae)
        result['mse'].append(mse)
    df_rst = pd.DataFrame.from_dict(result)
    return df_rst

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S') 
    df_rst = run_regressor(args)
    filtered_str = '' if args.raw else '_filtered'
    output_csv = f"rf_{args.llm}_{args.text}_prop_{len(props)}_{filtered_str}.csv"
    if args.output_dir:
        output_csv = os.path.join(args.output_dir, output_csv)
    df_rst.to_csv(output_csv)

