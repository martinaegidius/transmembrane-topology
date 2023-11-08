#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:04:44 2023

@author: max
"""

import os 
import numpy as np 
from collections import defaultdict
import pickle

def def_value():
    return "Not present"

#fileInName = "DeepTMHMM_crossval.top"
#fileOutName = "test"
#set_type="test"

def data_to_dicts(fileInName: str,fileOutName: str,set_type: str, savePickle: bool):
    #type: set to "train" or "test". Test-set does not contain type of protein
    #read 3.lines format for training the training-data, creating a dict with the protein name, the classification, the sequence, and the label
    path = "./data/"

    with open(path+fileInName) as f:
        data = f.readlines() #returns a complete list of the data 
        
    #order of entries in list: 
        #[0]: ">P53877|GLOBULAR\n" (ie. protein and type), 
        #[1]: Aminoacid sequence
        #[2]: Protein topology label
        #repeat...
    
    if(set_type=="train"):
        protein_strings = data[0::3] #every third element
        residues = data[1::3]
        topologies = data[2::3]
        
    else: 
        protein_strings = data[0::4] #every third element
        residues = data[1::4]
        topologies = data[2::4]
        topologies_second = data[3::4]
        
    
    #check that lengths match: success
        
    #remove ">" and "\n" in protein_strings
    protein_names_tmp = [x.replace(">","").replace("\n","") for x in protein_strings]
    protein_names = []
    if(set_type=="train"):
        protein_types = []
    for entry in protein_names_tmp:
        if(set_type=="train"):
            name_tmp, type_tmp = entry.split("|")
            protein_types.append(type_tmp)
        
        else:
            name_tmp = entry
        protein_names.append(name_tmp)
        
    #remove \n from residues and topologies
    residues = [x.replace("\n","") for x in residues]
    topologies = [x.replace("\n","") for x in topologies]
    if(set_type!="train"):
        topologies_second = [x.replace("\n","") for x in topologies_second]
    
    #send to default-dict, with a dict-of-dicts structure 
    
    d = defaultdict(def_value)
    
    for i, name in enumerate(protein_names):
        if(set_type=="train"):
            tmp_dict = {"residue_seq":residues[i],"topology_label":topologies[i],"protein_type":protein_types[i]}
        else:
            tmp_dict = {"residue_seq":residues[i],"topology_label":topologies[i],"topology_second_label":topologies_second[i]}
        d[name] = tmp_dict
        #[residues[i],topologies[i],protein_type[i]]
        
    #check that they are consistent with what is given in file using unit-test
    #idx_test = [0,17,662,-1]
    
    num_successes = []
    for idx in range(len(protein_strings)):
        test_name = protein_names[idx]
        test_sequence = residues[idx]
        test_topology = topologies[idx]
        if(set_type=="train"):
            test_type = protein_types[idx]
        dict_entry = d[test_name]
        name_check = dict_entry != "Not present"
        name_check2 = test_name in protein_strings[idx]
        sequence_check = dict_entry["residue_seq"] in test_sequence
        topology_check = dict_entry["topology_label"] in test_topology
        if(set_type=="train"):
            type_check = dict_entry["protein_type"] in protein_types[idx]
            type_check2 = dict_entry["protein_type"] in protein_strings[idx]
            test_result = name_check==True and name_check2==True and sequence_check == True and topology_check==True and type_check==True and type_check2==True
        else:
            topology_check2 = dict_entry["topology_second_label"] in topologies_second[idx]
            test_result = name_check==True and name_check2==True and sequence_check == True and topology_check==True and topology_check2 == True
            
        num_successes.append(test_result)
    
    assert sum(num_successes)==len(protein_strings), "Error: something seems to be wrong with processing. Aborting pickle"
    if(savePickle):
        print("Writing file to: ",path+fileOutName+".pkl")
        with open(path+fileOutName+".pkl", 'wb') as fp:
            pickle.dump(d,fp)
    
    return d

#data_to_dicts("DeepTMHMM_crossval.top","val","test")
#data_to_dicts("train.3line","train","train")
