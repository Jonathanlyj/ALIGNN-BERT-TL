#mean_absolute_error: 54.10120434782608
#mean_absolute_error_formation_energy_peratom: 0.471877877532833
import json
import numpy as np
import transformers

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import torch
from jarvis.core.atoms import Atoms
from jarvis.io.vasp.inputs import Poscar
from jarvis.db.figshare import data
from jarvis.io.vasp.inputs import Poscar
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from tqdm import tqdm 
import argparse
import logging
import pandas as pd
import os
# import chemnlp
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
from chemnlp.chemnlp.utils.describe import atoms_describer
from robocrys import StructureCondenser, StructureDescriber
print(transformers.__version__)
SEED = 1

parser = argparse.ArgumentParser(description='get structure files on dataset')

parser.add_argument('--output_dir', help='path to the save output embedding', default=None, type=str, required=False)
args,_ = parser.parse_known_args()




def generate_struc(args):
    output_dir = "./structure/"
    if args.output_dir:
        output_dir = args.output_dir
    dft_3d = data(dataset='dft_3d')
    for i in dft_3d[:]:
        atoms = Atoms.from_dict(i['atoms'])
        poscar = Poscar(atoms)
        jid = i['jid']
        filename = 'POSCAR-'+jid+'.vasp'
        filepath = os.path.join(output_dir, filename)
        poscar.write_file(filepath)
    return 




if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO) 
    generate_struc(args)
    logging.info(f"Finished generate structure files")

    

