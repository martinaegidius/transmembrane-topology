from proteinworkshop.features.factory import ProteinFeaturiser
from proteinworkshop.datasets.utils import create_example_batch
#from proteinworkshop.models.graph_encoders.schnet import SchNetModel
from torch.utils.data import Dataset
import pickle
#import pdbreader
from torch.utils.data import DataLoader
import transmembraneDataset
import transmembraneUtils as tmu
import torch
from transmembraneModels import GraphEncDec, init_linear_weights
import os 
import losswise

###adapted from google_drive_training_pipe in a modular fashion and to support DTU HPC

### ------------------ CFG FLAGS ----------------------
BATCH_SZ = 1
N_EPOCHS = 300
print_every = 20
eval_every = 10
DECODER = "LSTMB"
NUM_SAMPLES = 64
EXPERIMENTTYPE = "OVERFIT"
SPLITTYPE = "PROTOTYPE"
FEATURISERFLAG = "SIMPLE"
SAVERESULTS = True
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





torch.manual_seed(1)



#let's try generating the tensor data-set and see what happens
#pdb_dir = "data/graphein_downloads/" #downloaded using AFv4
pdb_dir = os.environ["BLACKHOLE"]+"/data/graphein_downloads/" #this gives /dtu/... may need to exclude first /
trainSet = transmembraneDataset(root=pdb_dir,setType="train",proteinlist=trainnames,labelDict=trainlabels,mismatching_proteins=mismatches)
valSet = transmembraneDataset(root=pdb_dir,setType="val",proteinlist=valnames,labelDict=vallabels,mismatching_proteins=mismatches)
testSet = transmembraneDataset(root=pdb_dir,setType="test",proteinlist=testnames,labelDict=testlabels,mismatching_proteins=mismatches)


trainSubset = torch.utils.data.Subset(trainSet,list(range(0,NUM_SAMPLES))) #logs in obsidian
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
trainloader = DataLoader(trainSubset,batch_size=BATCH_SZ,shuffle=False) #need to set true when finished overfitting!
valloader = DataLoader(valSet,batch_size=BATCH_SZ)
testloader = DataLoader(testSet,batch_size=BATCH_SZ)

param_cfg = {"experimentType": EXPERIMENTTYPE,"SplitType":SPLITTYPE,"NUM_TRAIN_SAMPLES":len(trainloader),"Featuriser":FEATURISERFLAG,"DecoderType":DECODER,"LR":1e-04,"GCSize":128,"LSTMSize":64,"Dropout":0.0,"BATCH_SZ":BATCH_SZ,"Number of epochs":N_EPOCHS,"Save results":SAVERESULTS}

type2key = {'I': 0, 'O':1, 'P': 2, 'S': 3, 'M':4, 'B': 5} 

if(FEATURISERFLAG=="SIMPLE"):
  featuriser = ProteinFeaturiser( #note: input is a protein Batch
          representation="CA",
          scalar_node_features=["amino_acid_one_hot"],
          vector_node_features=[],
          edge_types=["knn_16"],
          scalar_edge_features=["edge_distance"],
          vector_edge_features=[],
        )


###--------------------------- Start training -------------------------
losswise.set_api_key("KPFFS4CCK")
session = losswise.Session(params=param_cfg,max_iter=N_EPOCHS)
lossGraph = session.graph("loss",kind="min")
accGraph = session.graph("Accuracy",kind="max")
accGraphOverlap = session.graph("Overlap accuracy",kind="max")


model = GraphEncDec(featuriser=featuriser, n_classes=6,hidden_dim_GCN=param_cfg["GCSize"],decoderType=param_cfg["DecoderType"],LSTM_hidden_dim=param_cfg["LSTMSize"],dropout=param_cfg["Dropout"])
model.apply(init_linear_weights) #change to xavier normal init
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=param_cfg["LR"])
criterion = torch.nn.CrossEntropyLoss()
epoch_loss = []
#outputs = []
for i in range(N_EPOCHS):
    loss, output = tmu.train_single_epoch(model,optimizer,criterion,trainloader,type2key)
    epoch_loss.append(loss)
    #if(i%print_every==0):
    #    print(f"Loss in epoch {i}: {loss}")
    if(i%eval_every==0):
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

        valloss /= len(valloader)
        val_acc,val_acc_overlap = tmu.dataset_accuracy(model,valloader,type2key) #get accuracy

        train_acc,train_acc_overlap = tmu.dataset_accuracy(model,trainloader,type2key)

        lossGraph.append(i,{"train_loss":loss,"val_loss":valloss})
        accGraph.append(i,{"train_acc":train_acc,"val_acc":val_acc})
        accGraphOverlap.append(i,{"train_acc":train_acc_overlap,"val_acc":val_acc_overlap})

    else:
      lossGraph.append(i,{"train_loss":loss})

###TEST MODEL 
model.eval()
test_loss = 0.0
test_loss_results = {}
test_predictions = {}


with torch.no_grad():
  sm = torch.nn.Softmax(dim=1)
  for i, data in enumerate(testloader):
    output = model(data)
    label = tmu.label_to_tensor(data.label,type2key)
    loss = criterion(output,label) #no need to worry about batching as graph is disjoint by design -> in principle no batching
    test_loss += loss.item()
    test_loss_results[data.id[0].replace("_ABCD","")] = loss.item()
    preds_test = sm(output)
    preds_test = torch.argmax(preds_test,dim=1)
    label_string = tmu.tensor_to_label(preds_test,type2key)
    test_predictions[data.id[0].replace("_ABCD","")] = {"prediction":label_string,"label":data.label}
    
    
    

mean_test_loss = test_loss/len(testloader)
lossGraph.append(N_EPOCHS,{"test_loss":mean_test_loss})
#get accuracies across sets at end of experiment
train_acc,train_acc_overlap = tmu.dataset_accuracy(model,trainloader,type2key)
val_acc,val_acc_overlap = tmu.dataset_accuracy(model,valloader,type2key)
test_acc,test_acc_overlap = tmu.dataset_accuracy(model,testloader,type2key)
accGraph.append(N_EPOCHS,{"train_acc":train_acc,"val_acc":val_acc,"test_acc":test_acc})
accGraphOverlap.append(i,{"train_acc":train_acc_overlap,"val_acc":val_acc_overlap,"test_acc":test_acc_overlap})


sess_id = session.session_id
session.done()
root_dir = os.environ["BLACKHOLE"]
if(SAVERESULTS):
  tmu.saveResults(sess_id,root_dir,epoch_loss,val_loss_li,test_loss_results,test_predictions,train_acc,val_acc,test_acc,train_acc_overlap,val_acc_overlap,test_acc_overlap)

