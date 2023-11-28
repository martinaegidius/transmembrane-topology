import os 
import sys 
import pickle
import math
import string 
import random


BATCH_SZ = 32
LR = 1e-04
GCSIZE = [512]
LSTMSIZE = [512]
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
CLIPGRAD = False
TRACKING = False #TRUE WHEN RUNNING EXPERIMENTS WHICH SHOULD BE LOGGED
DEBUG = True
EARLYSTOPPING = True
LSTM_NORMALIZATION = True

NUM_SAMPLES = "ALL"
N_EPOCHS = 20


NEXP = len(LSTMSIZE)

for i in range(NEXP):
    cfg_id = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))
    param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":LR*math.sqrt(BATCH_SZ),"WEIGHTDECAY":WEIGHTDECAY,"GCSize":GCSIZE[i],"LSTMSize":LSTMSIZE[i],"Dropout":0.0,"BATCH_SZ":BATCH_SZ,"N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS,"SRC":SRC,"OPTIMSCHEDULE":OPTIMSCHEDULE,"SEED":SEED,"TRACKING":TRACKING,"DEBUG":DEBUG,"CLIPGRADS":CLIPGRAD,"EARLYSTOP":EARLYSTOPPING,"LSTMNORM":LSTM_NORMALIZATION}
    path = os.environ["BLACKHOLE"]+"/experimental_cfg"+"_"+cfg_id+".pkl"
    print(path)
    with open(path, "wb") as f:
        pickle.dump(param_cfg,f) #is overwritten at every iteration
    

    command = "python3 train_wandb.py " + str(path)
    os.system(command)
    



