import os 
import sys 
import pickle
import math
import string 
import random


BATCH_SZ = 32#
LR = 1e-04
GCSIZE = [128,256,2028]
LSTMSIZE = None
DROPOUT = 0.0
DECODER = "linear"
EXPERIMENTTYPE = "LINEAR DECODER"
SPLITTYPE = "PROTOTYPE"
FEATURISERFLAG = "INTERMEDIATE"
SAVERESULTS = True
WEIGHTDECAY = 0
SRC = "HPC CALLSCRIPT"
OPTIMSCHEDULE = "NONE"
CLIPGRAD = True
EARLYSTOPPING = False
SEED = 2
TRACKING = True #TRUE WHEN RUNNING EXPERIMENTS WHICH SHOULD BE LOGGED
DEBUG = False
CLIPGRAD = False
CLIPVAL = 1000
LSTM_NORMALIZATION = [True]
NUM_SAMPLES = "ALL"
N_EPOCHS = 140

NEXP = len(GCSIZE)

for i in range(NEXP):
    cfg_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    path = os.environ["BLACKHOLE"]+"/experimental_cfg"+"_"+cfg_id+".pkl"
    param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":LR*math.sqrt(BATCH_SZ),"WEIGHTDECAY":WEIGHTDECAY,"GCSize":GCSIZE[i],"LSTMSize":LSTMSIZE,"Dropout":0.0,"BATCH_SZ":BATCH_SZ,"N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS,"SRC":SRC,"OPTIMSCHEDULE":OPTIMSCHEDULE,"SEED":SEED,"TRACKING":TRACKING,"DEBUG":DEBUG,"CLIPGRAD":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"CLIPGRADS":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"CLIPVAL":CLIPVAL,"LSTMNORM":LSTM_NORMALIZATION[0]}
    with open(path, "wb") as f:
        pickle.dump(param_cfg,f) #is overwritten at every iteration
    

    command = "python3 train_wandb.py " + str(path)
    os.system(command)
    
