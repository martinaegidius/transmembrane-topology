#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 18:23:01 2023

@author: max
"""

from Bio import SeqIO
from data_to_dicts import data_to_dicts
from collections import defaultdict
import glob 
import pdbreader 
import pandas as pd

#dont use json - use pdb files 
#and biopython
#https://bioinformatics.stackexchange.com/questions/14101/extract-residue-sequence-from-pdb-file-in-biopython-but-open-to-recommendation
#https://mmcif.wwpdb.org/docs/sw-examples/python/html/fasta.html


train = data_to_dicts("train.3line","train","train",savePickle=False) #3576 samples is correct 
train_path = "data/graphein_downloads/train/"

#exclude non-found entries from train
with open(train_path+"missing/missing_prots.txt","r") as file:
    missing_prots = file.readlines()
    
missing_prots = [x.replace("\n","") for x in missing_prots]

#remove missing_prots from train dictionary 
for missing in missing_prots:
    if(missing in train):
        del train[missing]


def compare_residue_sequences(org_d,AF_path):
    """Compares downloaded alpha-fold residue-sequences to sequences provided in the train-set from DeepTMHMM
    Args: 
        org_d:   defaultdict object containing all training data excluding missing entries
        AF_path: the relative path of all alphafold structures downloaded
    Returns: 
        A dictionary containing protein-names which passed/failed test
    """
    
    protein_names = list(org_d.keys())

    translation_lex = {'CYS': 'C', 'ASP': 'D', 'SER': 'S', 'GLN': 'Q', 'LYS': 'K',
     'ILE': 'I', 'PRO': 'P', 'THR': 'T', 'PHE': 'F', 'ASN': 'N', 
     'GLY': 'G', 'HIS': 'H', 'LEU': 'L', 'ARG': 'R', 'TRP': 'W', 
     'ALA': 'A', 'VAL':'V', 'GLU': 'E', 'TYR': 'Y', 'MET': 'M'}    
    
    result_d = {"Equivalent sequence":[], "Different sequence":[]}

    for i, name in enumerate(protein_names):
        print(f"Checking {i}/{len(org_d)}")
        #SEQUENCE 
        train_seq = org_d[name]["residue_seq"]
        train_seq_df = pd.Series([*train_seq],name="resname")
        
        #AF-download
        pdb_path = AF_path+name+".pdb"
        pdb = pdbreader.read_pdb(pdb_path)
        #only interested in amino-acid sequence, so we drop duplicates in resid
        df = pdb["ATOM"]
        df = df.drop_duplicates('resid')
        AF_seq = df["resname"]
        
        #translate three-letter code to one-letter code
        AF_seq_translated = AF_seq.replace(translation_lex)
        AF_seq_translated = AF_seq_translated.reset_index(drop=True) #reset indices
        result = AF_seq_translated.equals(train_seq_df)
        if(result==True):
            result_d["Equivalent sequence"].append(name)
        else:
            result_d["Different sequence"].append(name)
        
    return result_d


quality_check = compare_residue_sequences(train,train_path)
import pickle 
with open("data_quality_control.pkl","wb") as f:
    pickle.dump(quality_check,f)




