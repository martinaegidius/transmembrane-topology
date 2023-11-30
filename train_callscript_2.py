import os 
import sys 
import pickle
import math
import string 
import random
import itertools


BATCH_SZ = [64]#[32,64]
LR = 9e-04
GCSIZE = [64]
LSTMSIZE = [128]
DROPOUT = [0.1]#[0.0,0.2]
DECODER = "LSTMB"
EXPERIMENTTYPE = "FITTING PACKED BROOK"
SPLITTYPE = "PROTOTYPE"
FEATURISERFLAG = "INTERMEDIATE"
SAVERESULTS = True
WEIGHTDECAY = 1e-04
SRC = "HPC"
OPTIMSCHEDULE = "NONE"
SEED = 2
CLIPGRAD = True
CLIPVAL = 5.0
TRACKING = True #TRUE WHEN RUNNING EXPERIMENTS WHICH SHOULD BE LOGGED
DEBUG = False
EARLYSTOPPING = True
NUM_SAMPLES = "ALL"
N_EPOCHS = 200
LSTM_NORMALIZATION = [True]
LSTM_LAYERS = [4]
VAL_EVERY = 2


for x in itertools.product(GCSIZE, LSTMSIZE,LSTM_NORMALIZATION,LSTM_LAYERS,BATCH_SZ,DROPOUT): #run through all combinations to check if layernorm and grad clip helps out
    gcsize = x[0]
    lstmsize = x[1]
    lstmnorm = x[2]
    lstmlayers = x[3]
    batch_sz = x[-2]
    dropout = x[-1]

    cfg_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":LR*math.sqrt(batch_sz),"WEIGHTDECAY":WEIGHTDECAY,"GCSize":gcsize,"LSTMSize":lstmsize,"Dropout":dropout,"BATCH_SZ":batch_sz,"N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS,"SRC":SRC,"OPTIMSCHEDULE":OPTIMSCHEDULE,"SEED":SEED,"TRACKING":TRACKING,"DEBUG":DEBUG,"CLIPGRADS":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"CLIPVAL":CLIPVAL,"LSTMNORM":lstmnorm,"LSTMLAYERS":lstmlayers,"VALINTERVAL":VAL_EVERY}
    path = os.environ["BLACKHOLE"]+"/experimental_cfg"+"_"+cfg_id+".pkl"
    with open(path, "wb") as f:
        pickle.dump(param_cfg,f) #is overwritten at every iteration
    

    command = "python3 train_wandb_packed.py " + str(path)
    os.system(command)
    



