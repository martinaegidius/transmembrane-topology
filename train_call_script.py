import os 
import sys 
import pickle
import math
import string 
import random
import itertools


BATCH_SZ = 32
LR = 1e-04
GCSIZE = [128,512]
LSTMSIZE = [128,512]
DROPOUT = 0.0
DECODER = "LSTMB"
EXPERIMENTTYPE = "FIT CLIPPED GRADS"
SPLITTYPE = "PROTOTYPE"
FEATURISERFLAG = "INTERMEDIATE"
SAVERESULTS = True
WEIGHTDECAY = 1e-04
SRC = "HPC CALLSCRIPT"
OPTIMSCHEDULE = "NONE"
SEED = 2
CLIPGRAD = True
CLIPVAL = 5.0
TRACKING = True #TRUE WHEN RUNNING EXPERIMENTS WHICH SHOULD BE LOGGED
DEBUG = False
EARLYSTOPPING = False
NUM_SAMPLES = "ALL"
N_EPOCHS = 140
LSTM_NORMALIZATION = [True]

NEXP = len(LSTMSIZE)

for x in itertools.product(GCSIZE, LSTMSIZE,LSTM_NORMALIZATION): #run through all combinations to check if layernorm and grad clip helps out
    gcsize = x[0]
    lstmsize = x[1]
    lstmnorm = x[-1]
    cfg_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":LR*math.sqrt(BATCH_SZ),"WEIGHTDECAY":WEIGHTDECAY,"GCSize":gcsize,"LSTMSize":lstmsize,"Dropout":0.0,"BATCH_SZ":BATCH_SZ,"N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS,"SRC":SRC,"OPTIMSCHEDULE":OPTIMSCHEDULE,"SEED":SEED,"TRACKING":TRACKING,"DEBUG":DEBUG,"CLIPGRADS":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"CLIPVAL":CLIPVAL,"LSTMNORM":lstmnorm}
    path = os.environ["BLACKHOLE"]+"/experimental_cfg"+"_"+cfg_id+".pkl"
    with open(path, "wb") as f:
        pickle.dump(param_cfg,f) #is overwritten at every iteration
    

    command = "python3 train_wandb.py " + str(path)
    os.system(command)
    



