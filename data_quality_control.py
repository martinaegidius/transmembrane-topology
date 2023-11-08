#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:11:32 2023

@author: max
"""

from data_to_dicts import data_to_dicts
from collections import defaultdict
from graphein.protein.utils import download_alphafold_structure
import glob 

#make preprocessing to dicts of dicts
train = data_to_dicts("train.3line","train","train",savePickle=False) #3576 samples is correct 
val = data_to_dicts("DeepTMHMM_crossval.top","val","test",savePickle=False) #3574 samples is correct
###todo: add protein type definition using hard-coding to the val-set when you have figured out why there are two topology-lines


###TODO: current 
#get alphafold labels
def download_matching_protein(protein_name,path):
    #protein_path = download_alphafold_structure(protein_name, out_dir = path, aligned_score=True)
    protein_path = download_alphafold_structure(protein_name, version=4,out_dir = path, aligned_score=True)
    if(protein_path==None):#case: download failed
        success = 0
    else:
        success = 1
    return [protein_path,success]
    
protein_names_train = list(train.keys())
protein_names_val = list(val.keys())

download_status_list = []
for i, name in enumerate(protein_names_train):
    print(f"Downloading {i}/{len(protein_names_train)}...")
    output = download_matching_protein(name,"data/graphein_downloads/train/")#"data/AF/"+protein_names_train[0])    
    download_status_list.append([name,output[-1]])
    if(i==4):
        print(download_status_list)
   
# there are two proteins only present in train, not present in val
# set(protein_names_train)-set(protein_names_val)
# {'A1JUB7', 'P02930'}
# there are no proteins in val which are not present in train. thus the following downloader is redundant.

"""for name in protein_names_val:
    output = download_matching_protein(name,"data/graphein_downloads/val/")#"data/AF/"+protein_names_train[0])    
    print("Downloaded ",output)
"""

#it seems we dont succeed getting all proteins. 
#lets check which ones we got, and which ones we didn't get
#note: graphein construct graph needs the relative path to the pdb-file
train_path = "data/graphein_downloads/train/"
val_path = "data/graphein_downloads/val/"
train_downloaded = glob.glob(train_path+"*.pdb",recursive=True)
val_downloaded = glob.glob(val_path+"*.pdb",recursive=True)

def strip_path(path_list : list):
    tmp = [x.rsplit("/",1)[-1] for x in path_list] #remove path
    return [x.split(sep=".")[0] for x in tmp] #remove .pdb -> result only protein name
    
train_downloaded_names = strip_path(train_downloaded)
val_downloaded_names = strip_path(val_downloaded)

print_no_missing = lambda download_li, org_li: print(f"Missing {len(org_li)-len(download_li)} entries")
print("train: \n")
print_no_missing(train_downloaded_names,protein_names_train)

print("val: \n")
print_no_missing(val_downloaded_names,protein_names_val)

def find_missing_proteins(download_li,org_li):
    ###Finds elements present in org_li and not present in download_li
    return set(org_li) - set(download_li)

missing_prots_train = find_missing_proteins(train_downloaded_names,protein_names_train)
missing_prots_val = find_missing_proteins(val_downloaded_names,protein_names_val)

equal_missing = missing_prots_train==missing_prots_val
print("Are all missing entries the same in train and val?: ",equal_missing)

#try to download missing entries using version 2 
missing_li = list(missing_prots_train) #they are equal, so we just do it once
for missing_prot in missing_li:
    protein_path = download_alphafold_structure(missing_prot, version = 2, out_dir = "data/graphein_downloads/train/v2_missing/", aligned_score=True)
    
#also not possible
#save to scratch for knowing which ones they are 
file = open("data/graphein_downloads/train/missing/missing_prots.txt",'w')
for item in missing_li:
    file.write(item+"\n")
file.close()
