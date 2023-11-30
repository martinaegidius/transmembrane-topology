print("STARTED SCRIPT")
from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch
#from proteinworkshop.models.graph_encoders.schnet import SchNetModel
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import pickle
from transmembraneDataset import transmembraneDataset
import transmembraneUtilsPacked as tmu
import torch
from transmembraneModelsPacked import GraphEncDec 
from transmembraneModelsPacked import init_linear_weights
import os 
from torch_geometric.utils import unbatch
import wandb
import numpy as np 
import torchmetrics
import copy


###adapted from google_drive_training_pipe in a modular fashion and to support DTU HPC
import sys
if(len(sys.argv)>1):
   cfg_path = sys.argv[-1]
   print("read path for cfg: ", cfg_path)
else:
   print("FOUND NO CFG!")
   cfg_path = None

print("Is cuda available?: ", torch.cuda.is_available())
torch.cuda.empty_cache()


### ------------------ CFG FLAGS ----------------------
if(cfg_path==None):
  BATCH_SZ = 1
  LR = 1e-04
  GCSIZE = 64
  LSTMSIZE = 64
  DROPOUT = 0.0
  N_EPOCHS = 200
  DECODER = "LSTMB"
  NUM_SAMPLES = 64
  EXPERIMENTTYPE = "OVERFIT"
  SPLITTYPE = "PROTOTYPE"
  FEATURISERFLAG = "SIMPLE"
  SAVERESULTS = True
  WEIGHTDECAY = 0.0
  param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_SAMPLES":NUM_SAMPLES,"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":LR,"WEIGHTDECAY":WEIGHTDECAY,"GCSize":GCSIZE,"LSTMSize":LSTMSIZE,"Dropout":0.0,"BATCH_SZ":BATCH_SZ,"N_EPOCHS":N_EPOCHS,"Save results":SAVERESULTS}

else:
  with open(cfg_path,"rb") as f:
    param_cfg = pickle.load(f)
    print(param_cfg)
  N_EPOCHS = param_cfg["N_EPOCHS"]
  SAVERESULTS = param_cfg["Save results"]
  NUM_SAMPLES = param_cfg["NUM_SAMPLES"]

try:
   eval_every = param_cfg["VALINTERVAL"]
   print_every = 1
except:
   if(param_cfg["TRACKING"]==True and param_cfg["DEBUG"]==False):
      print_every = 3#int(N_EPOCHS/60)
      eval_every = 5#int(N_EPOCHS/40)
   else: #case: debugging
      print_every = 1
      eval_every = 5

#print_every = 1
#eval_every = 2 
### ---------------------------------------------------


#note: this is prototype-split only
with open("splits/prototype/train.pkl",'rb') as f:
    train = pickle.load(f)

with open("splits/prototype/val.pkl",'rb') as f:
    val = pickle.load(f)

with open("splits/prototype/test.pkl",'rb') as f:
    test = pickle.load(f)

#exclude mismatching proteins
with open("data_quality_control.pkl","rb") as f:
    mismatches = pickle.load(f)


mismatches = mismatches["Different sequence"]
trainnames = [x["id"] for x in train]
testnames = [x["id"] for x in test]
valnames = [x["id"] for x in val]

##not used atm - may be used later
trainlabels = {}
for prot in train:
    trainlabels[prot["id"]]=prot["labels"]

vallabels = {}
for prot in val:
    vallabels[prot["id"]]=prot["labels"]

testlabels = {}
for prot in test:
    testlabels[prot["id"]]=prot["labels"]





torch.manual_seed(param_cfg["SEED"])



#let's try generating the tensor data-set and see what happens
#pdb_dir = "data/graphein_downloads/" #downloaded using AFv4
pdb_dir = os.environ["BLACKHOLE"]+"/data/graphein_downloads/" #this gives /dtu/... may need to exclude first /
trainSet = transmembraneDataset(root=pdb_dir,setType="train",proteinlist=trainnames,labelDict=trainlabels,mismatching_proteins=mismatches)
valSet = transmembraneDataset(root=pdb_dir,setType="val",proteinlist=valnames,labelDict=vallabels,mismatching_proteins=mismatches)
testSet = transmembraneDataset(root=pdb_dir,setType="test",proteinlist=testnames,labelDict=testlabels,mismatching_proteins=mismatches)

if(param_cfg["NUM_SAMPLES"]!="ALL"):
    print("DETECTED REQUEST FOR SUBSET OF DATA")
    subsetIdx = torch.randperm(len(trainSet))
    trainSet = torch.utils.data.Subset(trainSet,subsetIdx[1:param_cfg["NUM_SAMPLES"]+1]) ##for getting a decent-ish split
    subsetIdx = torch.randperm(len(valSet))
    valSet = torch.utils.data.Subset(valSet,subsetIdx[1:param_cfg["NUM_SAMPLES"]+1])
    subsetIdx = torch.randperm(len(testSet))
    testSet = torch.utils.data.Subset(testSet,subsetIdx[1:param_cfg["NUM_SAMPLES"]+1])
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


trainloader = DataLoader(trainSet,batch_size=param_cfg["BATCH_SZ"],shuffle=True,num_workers=0) #need to set true when finished overfitting!
valloader = DataLoader(valSet,batch_size=param_cfg["BATCH_SZ"],shuffle=False,num_workers=0)
testloader = DataLoader(testSet,batch_size=param_cfg["BATCH_SZ"],shuffle=False,num_workers=0)

type2key = {'I': 0, 'O':1, 'P': 2, 'S': 3, 'M':4, 'B': 5}  #theirs

if(param_cfg["Featuriser"]=="SIMPLE"):
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )

if(param_cfg["Featuriser"]=="INTERMEDIATE"):
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot","sequence_positional_encoding","alpha","kappa","dihedrals"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )
  
if(param_cfg["Featuriser"]=="COMPLEX"):#in general, according to paper, this should work less well than INTERMEDIATE
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot","sequence_positional_encoding","alpha","kappa","dihedrals","sidechain_torsions"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )



###--------------------------- Start training -------------------------
if(param_cfg["TRACKING"]==True):
  wandb.login(key="b84e040f3273aa091bbc451cf6e8ae81fa9b09f1")
  run = wandb.init(project=param_cfg["experimentType"],config=param_cfg)

model = GraphEncDec(featuriser=featuriser, n_classes=6,hidden_dim_GCN=param_cfg["GCSize"],decoderType=param_cfg["DecoderType"],LSTM_hidden_dim=param_cfg["LSTMSize"],dropout=param_cfg["Dropout"],LSTMnormalization = param_cfg["LSTMNORM"],lstm_layers=param_cfg["LSTMLAYERS"])
print(model)
model.apply(init_linear_weights) #change to xavier normal init
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=param_cfg["LR"],weight_decay=param_cfg["WEIGHTDECAY"])
criterion = torch.nn.CrossEntropyLoss()
if(param_cfg["OPTIMSCHEDULE"]!="NONE"):
   torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
epoch_loss = []

###MANAGE TRACKING OF MODEL WEIGHTS, PREDICTIONS AND LABELS
if(param_cfg["TRACKING"]==True):
   wandb.watch(model, log_freq=100)

#make a table hook to save predictions
def log_prediction_table(epoch,proteinName,label,prediction,accuracy,overlap_match,protein_label,protein_prediction,type):
    table = wandb.Table(columns=["epoch","protein","label","prediction","accuracy","overlap","type label","prediction label"])
    for epoch, name, label, pred, acc, overlap_match_, protein_lab,protein_pred in zip(epoch,proteinName,label,prediction,accuracy,overlap_match,protein_label,protein_prediction):
        table.add_data(epoch,name,label,pred,acc,overlap_match_,protein_lab,protein_pred)
    wandb.log({f"{type}/predictions_table":table},commit=False)


train_data_log = next(iter(trainloader)) #probably make a random sample subset instead at some point

gradclip = param_cfg["CLIPGRADS"]
if(gradclip):
   clipval = param_cfg["CLIPVAL"]
else:
   clipval = None


def gradNorm(model):
   grads = [
   param.grad.detach().flatten()
   for param in model.parameters()
   if param.grad is not None
   ]
   norm = torch.cat(grads).norm()
   return norm

class EarlyStopping:
    def __init__(self, tolerance=10,path="./"):
        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.min_loss = 1000.0
        self.modelCheckPoint = None
        self.path = path

    def __call__(self, validation_loss,model):
        print("Holding minimum loss: ", self.min_loss)
        print("received: ",validation_loss)
        print("with type: ",type(validation_loss))
        if(validation_loss<self.min_loss):
           print("ENCOUNTERED LOWER VALIDATION LOSS")
           self.modelCheckPoint = copy.deepcopy(model.state_dict())
           self.min_loss = validation_loss
           self.counter = 0
        else:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                print("EARLY STOPPING TRIGGERED.")
                self.saveModel()

    def saveModel(self):
       modPath = self.path+"checkpoint.pt"
       torch.save(self.modelCheckPoint, modPath)
       print("Saved best model checkpoint to: ", modPath)
       return 

if(param_cfg["EARLYSTOP"]):
   early_stopping = EarlyStopping(tolerance=15,path=os.environ["BLACKHOLE"]+"/"+run.name)


for i in range(param_cfg["N_EPOCHS"]):
    loss, output = tmu.train_single_epoch(model,optimizer,criterion,trainloader,type2key,gradclip,clipval)
    if(param_cfg["BATCH_SZ"]==1):
      loss /= len(trainSet)
    else:
      loss /= len(trainloader) #number of batches in loader, slightly positively biased if the number of batches is not equal
    epoch_loss.append(loss)
    
    grad_norm = gradNorm(model)
    train_metrics = {"train/loss":loss,
                     "train/epoch":i,
                     "train/Grad-L2":grad_norm}
    if(param_cfg["TRACKING"]==True):
       wandb.log(train_metrics)
    if(i%print_every==0):
      print(f"Loss in epoch {i}: {loss}")
      print(f"Gradient L2 norm in epoch {i} is {grad_norm}")
    
    if(i>0 and i%eval_every==0):
      val_loss_li = []
      valloss = 0.0
      model.eval()
      val_correct = 0
      val_incorrect = 0
      with torch.no_grad():
        for data in valloader:
          v_loss = 0.0
          pred,protein_lengths = model(data,data.batch)
          label = tmu.label_to_tensor(data.label,type2key)
          label = torch.split(label,protein_lengths)
          batch_sz = len(label)
          for pred_, label_ in zip(pred,label):
            v_loss += criterion(pred_,label_) #no need to worry about batching as graph is disjoint by design -> in principle no batching
          v_loss /= batch_sz #normalize loss so it is corresponding to length of the batch
          valloss += v_loss
          val_loss_li.append(v_loss.item())
      if(param_cfg["BATCH_SZ"]==1):
         valloss /= len(valSet)
      else:
         valloss /= len(valloader) #also normalize using the number of batches
      
      val_acc,val_acc_overlap = tmu.dataset_accuracy(model,valloader,type2key) #get accuracy
      print(f"Val accuracy/overlap/loss {val_acc}/{val_acc_overlap}/{valloss}")
      train_acc,train_acc_overlap = tmu.dataset_accuracy(model,trainloader,type2key)
      print(f"Train accuracy/overlap {train_acc}/{train_acc_overlap}")
      if(param_cfg["TRACKING"]==True):
        #get sample logs from VAL #after debugging this should maybe be wrapped in a function
        pred,protein_lengths = model.predict(data,data.batch) #gives a list 
        label = tmu.label_to_tensor(data.label,type2key)
        label = torch.split(label,protein_lengths)
        log_nsamples = len(pred) #always get for a single batch 
        epoch_ = [i for x in range(log_nsamples)] #repeat so it matches
        ids_, labels_, preds_, accuracy_, overlap_match_,label_type_,pred_type_= tmu.log_batch_elementwise_accuracies(pred,label,data.id)
        if(param_cfg["DEBUG"]):
          print(f"val: elementwise: {accuracy_}/{overlap_match_}")
        log_prediction_table(epoch_,ids_,labels_,preds_,accuracy_,overlap_match_,label_type_,pred_type_,type="val")

        #get sample logs from train 
        data = train_data_log #use same sample to not mess-up logging (else wandb believes another epoch has passed)
        #print("OUTPUT FROM NEXT TRAINLOADER ",data)
        pred, protein_lengths = model.predict(data,data.batch)
        label = tmu.label_to_tensor(data.label,type2key)
        label = torch.split(label,protein_lengths)
        
        #if(param_cfg["BATCH_SZ"]>1):
        #   pred_unbatched = list(unbatch(pred,data.batch))
        #else: 
        #   pred_unbatched = pred
        
        
        ids_, labels_, preds_, accuracy_, overlap_match_,label_type_,pred_type_ = tmu.log_batch_elementwise_accuracies(pred,label,data.id)
        if(param_cfg["DEBUG"]):
           print(f"train: elementwise: {accuracy_}/{overlap_match_}")
        log_prediction_table(epoch_,ids_,labels_,preds_,accuracy_,overlap_match_,label_type_,pred_type_,type="train")

        val_metrics = {"val/eval_epoch": i,
                    "val/loss":valloss,
                     "val/acc":val_acc,
                     "val/acc_overlap":val_acc_overlap}
      
        train_metrics = {"train/eval_epoch": i,
                         "train/acc":train_acc,
                     "train/acc_overlap":train_acc_overlap}
      
        wandb.log(val_metrics)
        wandb.log(train_metrics)


      print(f"Accuracies in epoch {i}: Train acc {train_acc}\t Train overlap acc {train_acc_overlap}")
      model.train()

      if(param_cfg["EARLYSTOP"]):
         early_stopping(valloss.item(),model)
         if early_stopping.early_stop:
            print("Early stopping and model from tolerance epochs ago saved to scratch. triggered at epoch :", i)
            del model 
            break


if(param_cfg["Featuriser"]=="SIMPLE"):
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )

if(param_cfg["Featuriser"]=="INTERMEDIATE"):
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot","sequence_positional_encoding","alpha","kappa","dihedrals"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )
  
if(param_cfg["Featuriser"]=="COMPLEX"):#in general, according to paper, this should work less well than INTERMEDIATE
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot","sequence_positional_encoding","alpha","kappa","dihedrals","sidechain_torsions"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )
    

if early_stopping.early_stop: 
   device = torch.device("cuda")
   model = GraphEncDec(featuriser=featuriser, n_classes=6,hidden_dim_GCN=param_cfg["GCSize"],decoderType=param_cfg["DecoderType"],LSTM_hidden_dim=param_cfg["LSTMSize"],dropout=param_cfg["Dropout"],LSTMnormalization = param_cfg["LSTMNORM"],lstm_layers=param_cfg["LSTMLAYERS"])
   model.load_state_dict(torch.load(early_stopping.path+"checkpoint.pt"))
   model = model.to(device)
   #model.to(device)
   model.eval()
   print("Loaded best saved model")

else:
   torch.save(model.state_dict(), os.environ["BLACKHOLE"]+"/"+run.name+"checkpoint.pt")
   print("Early stopping never encountered. Saved current state to scratch.")
   


###-----------------------------------------------TEST MODEL 
model.eval()
test_loss = 0.0
test_loss_results = {}
test_predictions = {}




with torch.no_grad():
  sm = torch.nn.Softmax(dim=1)
  if(param_cfg["TRACKING"]==True): 
    test_sample_idx = [int(np.random.random()*param_cfg["BATCH_SZ"])] #choose which batches to log to table
  for i, data in enumerate(testloader): #always evaluated as single batch - maybe should be fixed
    loss = 0.0
    preds,protein_lengths = model(data,data.batch)
    label = tmu.label_to_tensor(data.label,type2key) #numeric conversion
    label = torch.split(label,protein_lengths) #list of numeric labels
    pred_classes = []
    for pred_, label_,id_,labelStr_ in zip(preds,label,data.id,data.label):
       loss_ = criterion(pred_,label_) #no need to worry about batching as graph is disjoint by design -> in principle no batching
       loss += loss_
       pred_sm = sm(pred_)
       pred_test_ = torch.argmax(pred_sm,dim=1)
       pred_classes.append(pred_test_)   

       predStr = tmu.tensor_to_label(pred_test_,type2key)

       proteinTypePred,pred_protein_label = tmu.type_from_labels(pred_test_)
       proteinTypeLabel,true_protein_label = tmu.type_from_labels(label_)
          #print("shooting into loss: ")
          #print(unbatched_probs[j])
          #print(labelU)
          ###add in matching aswell 
          #get topology 
       preds_top = tmu.label_list_to_topology(pred_test_)
       label_top = tmu.label_list_to_topology(label_)
       match_tmp = tmu.is_topologies_equal(label_top,preds_top)

       #print("IN EVAL TEST: ")
       #print("prediction")
       #print(pred_test_)
       #print("label")
       #print(label_)


       test_loss_results[id_.replace("_ABCD","")] = loss_.item()
       test_predictions[id_.replace("_ABCD","")] = {"prediction":pred_test_.tolist(),"label":label_.tolist(),"Type prediction":pred_protein_label,"Type label":true_protein_label,"Match":match_tmp}   
    loss/=len(preds) #normalize for batch
    test_loss += loss.item()
    if(param_cfg["TRACKING"]==True):
        if i in test_sample_idx: #if this batch is to be saved to table
            epoch_ = [param_cfg["N_EPOCHS"] for x in range(len(pred_classes))]#repeat so it matches
            ids_, labels_, preds_, accuracy_, overlap_match_,label_type_,pred_type_ = tmu.log_batch_elementwise_accuracies(pred_classes,label,data.id)
            log_prediction_table(epoch_,ids_,labels_,preds_,accuracy_,overlap_match_,label_type_,pred_type_,type="test")


if(param_cfg["BATCH_SZ"]==1):
   mean_test_loss = test_loss/len(testSet)
else:
   mean_test_loss = test_loss/len(testloader) #batches 

#get accuracies across sets at end of experiment
print("Evaluating train")
train_acc,train_acc_overlap,train_confmat_pr,train_confmat_type,train_AUROC,MCROC_train,trainMetrics = tmu.dataset_accuracy(model,trainloader,type2key,metrics=True)
print("Evaluating val")
val_acc,val_acc_overlap,val_confmat_pr,val_confmat_type,val_AUROC,MCROC_val,valMetrics = tmu.dataset_accuracy(model,valloader,type2key,metrics=True)
print("Evaluating test")
test_acc,test_acc_overlap,test_confmat_pr,test_confmat_type,test_AUROC,MCROC_test, testMetrics = tmu.dataset_accuracy(model,testloader,type2key,metrics=True)
if(param_cfg["TRACKING"]==True):  
   summary_titles = ["f_test_Oacc","f_val_Oacc","f_train_Oacc","f_test_acc","f_val_acc","f_train_acc"]
   #["f_train_acc","f_train_Oacc","f_val_acc","f_val_Oacc","f_test_acc","f_test_Oacc"]  
   #endresults = [train_acc,train_acc_overlap,val_acc,val_acc_overlap,test_acc,test_acc_overlap]
   endresults = [test_acc_overlap,val_acc_overlap,train_acc_overlap,test_acc,val_acc,train_acc]
   for i, name in enumerate(summary_titles):
      wandb.summary[name] = endresults[i]
   for k, v in testMetrics.items():
      wandb.summary[k] = v



#PLOT MULTICLASS ROC CURVES TO WANDB
if(param_cfg["TRACKING"]):
   figtest_, axtest_ = MCROC_test.plot(score=True)
   wandb.log({"test/ROC":wandb.Image(figtest_)})
   del figtest_, axtest_ 

   figtrain_, axtrain_ = MCROC_train.plot(score=True)
   wandb.log({"train/ROC":wandb.Image(figtrain_)})
   del figtrain_, axtrain_ 
   figval_, axval_ = MCROC_val.plot(score=True)
   wandb.log({"val/ROC":wandb.Image(figval_)})
   del figval_, axval_






print("------------------- PR CONFUSION MATRICES TEST-SET----------------------")
print(test_confmat_pr)
print("------------------- TYPE CONFUSION MATRICES TEST-SET----------------------")
print(test_confmat_type)
print("------------------- AUROC TEST-SET----------------------")
print(test_AUROC)
print("------------------- PR CONFUSION MATRICES VAL-SET----------------------")
print(val_confmat_pr)
print("------------------- TYPE CONFUSION MATRICES VAL-SET----------------------")
print(val_confmat_type)
print("------------------- AUROC VAL-SET----------------------")
print(val_AUROC)
print("------------------- PR CONFUSION MATRICES TRAIN-SET----------------------")
print(train_confmat_pr)
print("------------------- TYPE CONFUSION MATRICES TRAIN-SET----------------------")
print(train_confmat_type)
print("------------------- AUROC TRAIN-SET----------------------")
print(train_AUROC)


print("---------------------- ACCURACY METRICS ---------------------")
print("Train acc: ",train_acc)
print("Train acc overlap: ",train_acc_overlap)
print("Val acc: ",val_acc)
print("Val acc overlap: ",val_acc_overlap)
print("Test acc: ",test_acc)
print("Test acc overlap: ",test_acc_overlap)



if(param_cfg["TRACKING"]==True):    
  sess_id = run.name
  root_dir = os.environ["BLACKHOLE"]
  if(SAVERESULTS):
    tmu.saveResults(sess_id,root_dir,epoch_loss,val_loss_li,test_loss_results,test_predictions,train_acc,val_acc,test_acc,train_acc_overlap,val_acc_overlap,test_acc_overlap,param_cfg)
  wandb.finish()







