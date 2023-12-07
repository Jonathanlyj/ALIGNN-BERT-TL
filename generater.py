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
parser.add_argument('--text', help='text sources for sample', choices=['raw', 'chemnlp', 'robo'],default='raw', type=str, required=False)
args,_ = parser.parse_known_args()

def describe_chemical_data(data):
    description = ""

    if 'chemical_info' in data:
        description += "The chemical information include: "
        chem_info = data['chemical_info']
        description += f"The chemical has an atomic formula of {chem_info.get('atomic_formula', 'N/A')} with a prototype of {chem_info.get('prototype', 'N/A')};"
        description += f"Its molecular weight is {chem_info.get('molecular_weight', 'N/A')} g/mol; "
        description += f"The atomic fractions are {chem_info.get('atomic_fraction', 'N/A')}, and the atomic values X and Z are {chem_info.get('atomic_X', 'N/A')} and {chem_info.get('atomic_Z', 'N/A')}, respectively."

    if 'structure_info' in data:
        description += "The structure information include: "
        struct_info = data['structure_info']
        description += f"The lattice parameters are {struct_info.get('lattice_parameters', 'N/A')} with angles {struct_info.get('lattice_angles', 'N/A')} degrees; "
        description += f"The space group number is {struct_info.get('spg_number', 'N/A')} with the symbol {struct_info.get('spg_symbol', 'N/A')}; "
        description += f"The top K XRD peaks are found at {struct_info.get('top_k_xrd_peaks', 'N/A')} degrees; "
        description += f"The material has a density of {struct_info.get('density', 'N/A')} g/cm³, crystallizes in a {struct_info.get('crystal_system', 'N/A')} system, and has a point group of {struct_info.get('point_group', 'N/A')}; "
        description += f"The Wyckoff positions are {struct_info.get('wyckoff', 'N/A')}; "
        description += f"The number of atoms in the primitive and conventional cells are {struct_info.get('natoms_primitive', 'N/A')} and {struct_info.get('natoms_conventional', 'N/A')}, respectively; "
        
        if 'bond_distances' in struct_info:
            bond_distances = struct_info['bond_distances']
            bond_descriptions = ", ".join([f"{bond}: {distance} " for bond, distance in bond_distances.items()])
            description += f"The bond distances are as follows: {bond_descriptions}. "
    return description.strip()



def get_robo(structure=None):
    describer = StructureDescriber()
    condenser = StructureCondenser()
    condensed_structure = condenser.condense_structure(structure)
    description = describer.describe(condensed_structure)
    return description

def get_text(atoms, text):
    if text == 'robo':
        return get_robo(atoms.pymatgen_converter())
    elif text == 'raw':
        return Poscar(atoms).to_string()
    elif text == "chemnlp":
        return describe_chemical_data(atoms_describer(atoms=atoms))


def main(args):
    dat = data('dft_3d')
    text_dic = defaultdict(list)
    err_ct = 0
    end = len(dat)
    if args.end:
        end = min(args.end, len(dat))
    for entry in tqdm(dat[args.start:end], desc="Processing data"):
        try:
            text = get_text(Atoms.from_dict(entry['atoms']), args.text)
            text_dic['jid'].append(entry['jid'])
            text_dic['formula'].append(entry['formula'])
            text_dic['text'].append(text)
        except Exception as enumerate:
            err_ct += 1
            logging.info(f"Failed text generation count:{err_ct}")
    df_text = pd.DataFrame.from_dict(text_dic)
    output_file = f"{args.text}_{args.start}_{end}.csv"
    if args.output_dir:
        output_file = os.path.join(args.output_dir, output_file)
    df_text.to_csv(output_file)
    logging.info(f"Saved output text to {output_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    main(args)
    logging.info(f"Finished generate text")