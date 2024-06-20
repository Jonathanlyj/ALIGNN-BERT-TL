# ALIGNN-BERT-TL

This repository contains the code for Hybrid-LLM-GNN transfer learning framework to predict materials properties using concatenated embeddings extracted from ALIGNN and BERT models. The code provides the following functions:

* Generate string representations of provided crystal samples using NLP tools: Robocystallographer and ChemNLP
* Use a pre-trained BERT/MatBERT langugae model to extract context-aware word embeddings from text descriptions of a given crystal structure
* Concatenate LLM embeddings with ALIGNN embeddings obtained from [ALIGNN transfer learning pipeline](https://github.com/NU-CUCIS/ALIGNNTL/tree/main)
* Post-prediction analysis script for prediction peroformance analysis and text-based model explanation

## Installation 

The basic requirement for using the files are a Python 3.8 with the packages listed in requirements.txt. It is advisable to create an virtual environment with the correct dependencies.

The work related experiments was performed on Linux Fedora 7.9 Maipo.

## Source Files

