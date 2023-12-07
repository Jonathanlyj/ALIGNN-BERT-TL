#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model, BertModel, OPTModel
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
import pandas as pd
import os
# import chemnlp
from chemnlp.chemnlp.utils.describe import atoms_describer
from robocrys import StructureCondenser, StructureDescriber

SEED = 1

parser = argparse.ArgumentParser(description='get embeddings on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--label', help='output variable', default=None, type=str,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'],default='raw', type=str, required=False)
parser.add_argument('--llm', help='pre-trained llm to use', default='gpt2', type=str,required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
parser.add_argument('--cache_csv', help='path that stores text', default=None, type=str, required=False)
parser.add_argument('--cpu', action='store_true', help='use cpu only', required=False)
args,_ = parser.parse_known_args()




def get_robo(structure=None):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    return description

def preprocess_data(args):
    dat = data('dft_3d')
    dd = []
    if args.label:
        prop = args.label
        for i in dat:
            if i[prop]!='na': 
                dd.append(i)
        logging.info(f"Filter based on {prop} property: {len(dd)} samples left")
    else:
        dd = dat
        logging.info(f"Use full dataset: {len(dd)} samples")
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = args.llm
    tokenizer = AutoTokenizer.from_pretrained(llm)
    #model = AutoModelForSequenceClassification.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    if "gpt2" in llm.lower():
        model = GPT2Model.from_pretrained(llm)
    elif "bert" in llm.lower():
        model = BertModel.from_pretrained(llm)
    elif "opt" in llm.lower():
        model = OPTModel.from_pretrained(llm)

    model.to(device)
    embeddings = []
    samples=[]
    # print(model)
    max_token_length = model.config.max_position_embeddings
    logging.info(f"Max token length: {max_token_length}")
    if args.cache_csv:
        df_text = pd.read_csv(args.cache_csv, index_col = 'jid')
    for entry in tqdm(dat, desc="Processing data"):
        if args.cache_csv:
            if entry['jid'] in df_text.index:
                text = df_text.at[entry['jid'],'text']
            else:
                continue
        else:
            if args.text == 'raw':
                text = Poscar(Atoms.from_dict(entry['atoms'])).to_string()
            elif args.text == 'chemnlp':
                text = json.dumps(atoms_describer(atoms=Atoms.from_dict(entry['atoms'])))
            elif args.text == "robo":
                try:
                    text = get_robo(Atoms.from_dict(entry['atoms']).pymatgen_converter())
                except Exception as exp: 
                    pass
        inputs = tokenizer(text, return_tensors="pt").to(device)
        if len(inputs['input_ids'][0]) <= max_token_length:
            with torch.no_grad():
                output = model(**inputs)
            if device.type == 'cuda':
                emb = output.last_hidden_state.mean(dim=1).cpu().numpy().flatten()
            else:
                emb = output.last_hidden_state.mean(dim=1).numpy().flatten()
            embeddings.append(emb)
            samples.append(entry['jid'])


    embeddings = np.vstack(embeddings)
    #labels = np.array([entry['exfoliation_energy'] for entry in dat])

    return embeddings, samples

def save_data(embeddings, samples):
    n = len(embeddings)
    assert n == len(samples)
    df = pd.DataFrame(embeddings, index=samples)
    file_path = f"embeddings_{args.llm}_{args.text}_{n}.csv"
    if args.output_dir:
        file_path = os.path.join(args.output_dir, file_path)  
    df.to_csv(file_path)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    embeddings, samples = preprocess_data(args)
    logging.info(f"Finished generate embeddings")
    save_data(embeddings, samples)

