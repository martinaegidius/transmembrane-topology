from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch
#from proteinworkshop.models.graph_encoders.schnet import SchNetModel
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import pickle
from transmembraneDataset import transmembraneDataset
import transmembraneUtils as tmu
import torch
from transmembraneModels import GraphEncDec 
from transmembraneModels import init_linear_weights
import os 
import losswise
from torch_geometric.utils import unbatch

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


if(param_cfg["TRACKING"]==True):
  print_every = int(N_EPOCHS/20)
  eval_every = int(N_EPOCHS/20)
else: #case: debugging
   print_every = 1
   eval_every = 2
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

#print(type(trainSet))

subsetIdx = torch.randperm(len(trainSet))
#print("Subset: ",subsetIdx[:param_cfg["NUM_SAMPLES"]]) #for getting a decent-ish split
trainSubset = torch.utils.data.Subset(trainSet,subsetIdx[1:param_cfg["NUM_SAMPLES"]+1]) ##for getting a decent-ish split
#valSubset = torch.utils.data.Subset(valSet,list(range(0,4)))
#testSubset = torch.utils.data.Subset(testSet,list(range(0,4)))
#valSet = torch.utils.data.Subset(valSet,list(range(0,2)))
#testSet = torch.utils.data.Subset(testSet,list(range(0,2)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#valloader = DataLoader(valSubset,batch_size=BATCH_SZ,shuffle=False) #need to set true when finished overfitting!
#testloader = DataLoader(testSubset,batch_size=BATCH_SZ,shuffle=False) #need to set true when finished overfitting!


trainloader = DataLoader(trainSubset,batch_size=param_cfg["BATCH_SZ"],shuffle=True) #need to set true when finished overfitting!
#for i, data in enumerate(trainloader):
#   print(data.label)


valloader = DataLoader(valSet,batch_size=param_cfg["BATCH_SZ"],shuffle=False)
testloader = DataLoader(testSet,batch_size=param_cfg["BATCH_SZ"],shuffle=False) #always evaluate single batch



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

# for i, data in enumerate(trainloader):
#     print(i)
#     print(data)


###--------------------------- Start training -------------------------
if(param_cfg["TRACKING"]==True):
  losswise.set_api_key("KPFFS4CCK")
  print(param_cfg)
  session = losswise.Session(tag="HPC",max_iter=param_cfg["N_EPOCHS"],params=param_cfg,track_git=False)
  lossGraph = session.graph("loss",kind="min")
  accGraph = session.graph("Accuracy",kind="max")
  accGraphOverlap = session.graph("Overlap accuracy",kind="max")



model = GraphEncDec(featuriser=featuriser, n_classes=6,hidden_dim_GCN=param_cfg["GCSize"],decoderType=param_cfg["DecoderType"],LSTM_hidden_dim=param_cfg["LSTMSize"],dropout=param_cfg["Dropout"])
#if(param_cfg["DecoderType"]=="Linear"):
model.apply(init_linear_weights) #change to xavier normal init
model = model.to(device)
#print("Initialization predictions: ")
#tmu.dataset_accuracy(model,trainloader,type2key,mode="PRINT")
optimizer = torch.optim.Adam(model.parameters(),lr=param_cfg["LR"],weight_decay=param_cfg["WEIGHTDECAY"])
criterion = torch.nn.CrossEntropyLoss()
if(param_cfg["OPTIMSCHEDULE"]!="NONE"):
   torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
epoch_loss = []
#outputs = []

for i in range(param_cfg["N_EPOCHS"]):
    loss, output = tmu.train_single_epoch(model,optimizer,criterion,trainloader,type2key)
    loss /= len(trainSubset)
    epoch_loss.append(loss)
    if(i%print_every==0):
      print(f"Loss in epoch {i}: {loss}")
    if(i>0 and i%eval_every==0):
      val_loss_li = []
      valloss = 0.0
      model.eval()
      val_correct = 0
      val_incorrect = 0
      with torch.no_grad():
        for data in valloader:
          pred = model(data)
          label = tmu.label_to_tensor(data.label,type2key)
          #print(pred.shape)
          #print(label.shape)
          #print("label in val: ",label)
          #print("with type: ",type(label))
          v_loss = criterion(pred,label)
          valloss += v_loss
          val_loss_li.append(v_loss.item())

      valloss /= len(valSet)
      val_acc,val_acc_overlap = tmu.dataset_accuracy(model,valloader,type2key) #get accuracy
      train_acc,train_acc_overlap = tmu.dataset_accuracy(model,trainloader,type2key)
      print(f"Accuracies in epoch {i}: Train acc {train_acc}\t Train overlap acc {train_acc_overlap}")
      if(param_cfg["TRACKING"]==True):
        lossGraph.append(i,{"train_loss":loss,"val_loss":valloss})
        accGraph.append(i,{"train_acc":train_acc,"val_acc":val_acc})
        accGraphOverlap.append(i,{"train_acc":train_acc_overlap,"val_acc":val_acc_overlap})
      model.train()

    else:
      if(param_cfg["TRACKING"]==True):
        lossGraph.append(i,{"train_loss":loss})

###-----------------------------------------------TEST MODEL 
model.eval()
test_loss = 0.0
test_loss_results = {}
test_predictions = {}


with torch.no_grad():
  sm = torch.nn.Softmax(dim=1)
  for i, data in enumerate(testloader):
    output = model(data)
    label_f = tmu.label_to_tensor(data.label,type2key)
    loss = criterion(output,label_f) #no need to worry about batching as graph is disjoint by design -> in principle no batching
    test_loss += loss.item()
    preds_sm = sm(output)
    preds_test = torch.argmax(preds_sm,dim=1)
    #we still want to save per-protein predictions, so we unbatch
    if(len(data.label)>1): #case: is batched
       unbatched_preds = unbatch(preds_test,data.batch)
       unbatched_probs = unbatch(preds_sm,data.batch)
       for j, pred_label in enumerate(unbatched_preds): #save predictions
          labelStr = tmu.tensor_to_label(pred_label,type2key)
          labelU = tmu.label_to_tensor(data.label[j],type2key)
          #print("shooting into loss: ")
          #print(unbatched_probs[j])
          #print(labelU)
          loss_tmp = criterion(unbatched_probs[j],labelU)
          test_loss_results[data.id[j].replace("_ABCD","")] = loss_tmp.item()
          test_predictions[data.id[j].replace("_ABCD","")] = {"prediction":labelStr,"label":data.label[j]}
          #if(i==710):
          #  print("Saving string: ", labelStr)
          #  print("Saving label: ",data.label[j])
    else:
      test_loss_results[data.id[0].replace("_ABCD","")] = loss.item()
      label_string = tmu.tensor_to_label(preds_test,type2key)
      test_predictions[data.id[0].replace("_ABCD","")] = {"prediction":label_string,"label":data.label}


mean_test_loss = test_loss/len(testSet)
#get accuracies across sets at end of experiment
train_acc,train_acc_overlap = tmu.dataset_accuracy(model,trainloader,type2key)
val_acc,val_acc_overlap = tmu.dataset_accuracy(model,valloader,type2key)
test_acc,test_acc_overlap = tmu.dataset_accuracy(model,testloader,type2key)
if(param_cfg["TRACKING"]==True):    
  lossGraph.append(param_cfg["N_EPOCHS"],{"test_loss":mean_test_loss})
  accGraph.append(param_cfg["N_EPOCHS"],{"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc})
  accGraphOverlap.append(param_cfg["N_EPOCHS"],{"train_acc":train_acc_overlap,"val_acc":val_acc_overlap,"test_acc":test_acc_overlap})

print("Train acc: ",train_acc)
print("Train acc overlap: ",train_acc_overlap)
print("Val acc: ",val_acc)
print("Val acc overlap: ",val_acc_overlap)
print("Test acc: ",test_acc)
print("Test acc overlap: ",test_acc_overlap)


#print("test predictions are: ")
#print(test_predictions)
if(param_cfg["TRACKING"]==True):    
  sess_id = session.session_id
  session.done()
  root_dir = os.environ["BLACKHOLE"]
  if(SAVERESULTS):
    tmu.saveResults(sess_id,root_dir,epoch_loss,val_loss_li,test_loss_results,test_predictions,train_acc,val_acc,test_acc,train_acc_overlap,val_acc_overlap,test_acc_overlap,param_cfg)

