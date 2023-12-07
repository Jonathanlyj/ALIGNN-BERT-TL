import json
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, GPT2Model, BertModel
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
import warnings
from collections import defaultdict
import logging


warnings.filterwarnings("ignore", category=UserWarning)
parser = argparse.ArgumentParser(description='get embeddings on dataset')
# parser.add_argument('--data_path', help='path to the dataset',default=None, type=str, required=False)
parser.add_argument('--start', default=0, type=int,required=False)
# parser.add_argument('--input', help='input attributes set', default=None, type=str, required=False)
parser.add_argument('--end', type=int, required=False)
parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
args,_ = parser.parse_known_args()
def get_robo(structure=None):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    return description



def main(args):
    dat = data('dft_3d')
    robo_dic = defaultdict(list)
    robo_err_ct = 0
    end = len(dat)
    if args.end:
        end = min(args.end, len(dat))
    for entry in tqdm(dat[args.start:end], desc="Processing data"):
        try:
            text = get_robo(Atoms.from_dict(entry['atoms']).pymatgen_converter())
            robo_dic['jid'].append(entry['jid'])
            robo_dic['formula'].append(entry['formula'])
            robo_dic['text'].append(text)
        except Exception as enumerate:
            robo_err_ct += 1
            logging.info(f"Failed text generation count:{robo_err_ct}")
    df_robo = pd.DataFrame.from_dict(robo_dic)
    output_file = f"robo_{args.start}_{end}.csv"
    if args.output_dir:
        output_file = os.path.join(args.output_dir, output_file)
    df_robo.to_csv(output_file)


# dd=dd[0:100]
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    main(args)
    logging.info(f"Finished generate robo text")