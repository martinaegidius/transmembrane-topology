
import torch
import json 
import os


###scripts for label-conversion both-ways 
def label_to_tensor(labels,type2key):
    """
    Encodes corresponding numeric labels to residue cell-position
    Input:
        label: a list. Length of list indicates the number of samples in the batch
    """
    #unbatch by creating large string
    collated_str = ""
    for protein_label in labels:
        collated_str += protein_label

    collated_char_li = [*collated_str]
    #print(chars)
    default = "UNK"
    output = [type2key.get(x, default) for x in collated_char_li]
    output = torch.as_tensor(output).to('cuda:0')
    #from torch.nn.functional import one_hot
    #return one_hot(output,num_classes = len(type2key))
    return output


def tensor_to_label(labels,type2key):
  """Reversed functionality of label_to_tensor - maps one-hot predictions to sequence string"""
  key2type = dict((v,k) for k,v in type2key.items())
  labels = labels.tolist()
  outStr = ""
  for label in labels:
    outStr += key2type[label]

  return outStr


def saveResults(session_id,root_path,train,val,test,test_predictions,train_acc,val_acc,test_acc,train_acc_overlap,val_acc_overlap,test_acc_overlap):
  pwd_tmp = os.path.join(root_path,"ExperimentalResults")
  print(pwd_tmp)
  if not (os.path.exists(pwd_tmp)):
    os.mkdir(pwd_tmp)
    print("Constructed result directory")
  
  pwd_tmp = os.path.join(pwd_tmp,session_id)
  if not (os.path.exists(pwd_tmp)):
    os.mkdir(pwd_tmp) #make dir for session id - this is unique, so needs to made every time
  with open(pwd_tmp+"/train.txt", "w") as f:
    for line in train:
      f.write(str(line)+"\n")
  
  with open(pwd_tmp+"/val.txt", "w") as f:
    for line in val:
      f.write(str(line)+"\n")
  
  with open(pwd_tmp+"/test.txt","w") as f: 
    f.write(json.dumps(test))

  accs = {"Train accuracy":train_acc,"Train accuracy overlap":train_acc_overlap,"Validation accuracy":val_acc,"Validation accuracy overlap":val_acc_overlap,"Test accuracy":test_acc,"Test accuracy overlap":test_acc_overlap}
  with open(pwd_tmp+"/accuracies.txt","w") as f:
    f.write(json.dumps(accs))
    
  with open(pwd_tmp+"/test_predictions.txt","w") as f:
    f.write(json.dumps(test_predictions))
  
  
  return None


def dataset_accuracy(model,dataloader,type2key):
    """Calculates the per-residue topology accuracy"""
    model.eval()
    correct_single = 0
    incorrect_single = 0
    matching_overlap = 0
    total = 0
    for i, data in enumerate(dataloader):
        preds = model.predict(data)
        label = label_to_tensor(data.label,type2key)
        #single residue accuracies
        correct_single += torch.sum((preds==label))
        incorrect_single += torch.sum((preds!=label))
        #overlap-criterion accuracies
        preds = label_list_to_topology(preds)
        label = label_list_to_topology(label)
        matching_overlap += sequence_equality(preds,label)
        total += 1
    
    overlap_accuracy = matching_overlap/total
    single_residue_accuracy = (correct_single/(correct_single+incorrect_single)).sum()



    return single_residue_accuracy, overlap_accuracy

def label_list_to_topology(labels: torch.Tensor):
    """
    Converts a list of per-position labels to a topology representation.
    This maps every sequence to list of where each new symbol start (the topology), e.g. AAABBBBCCC -> [(0,A),(3, B)(7,C)]

    Parameters
    ----------
    labels : list or torch.Tensor of ints
        List of labels.

    Returns
    -------
    list of torch.Tensor
        List of tensors that represents the topology.
    """

    if isinstance(labels, list):
        labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
        zero_tensor = torch.LongTensor([0])
        if labels.is_cuda:
            zero_tensor = zero_tensor.cuda()

        unique, count = torch.unique_consecutive(labels, return_counts=True)
        top_list = [torch.cat((zero_tensor, labels[0:1]))]
        prev_count = 0
        i = 0
        for _ in unique.split(1):
            if i == 0:
                i += 1
                continue
            prev_count += count[i - 1]
            top_list.append(torch.cat((prev_count.view(1), unique[i].view(1))))
            i += 1
        return top_list
    

def sequence_equality(topology_a,topology_b,minimum_segment_overlap=5):
    if isinstance(topology_a[0], torch.Tensor):
            print("Converted a to list!")
            topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
            print("Converted b to list!")
            topology_b = list([b.cpu().numpy() for b in topology_b])
    if len(topology_a) != len(topology_b):
            return False
    
    for idx, (pos_a,label_a) in enumerate(topology_a):
        #print("checking: ",idx)
        pos_b, label_b = topology_b[idx]
        #print("Pos a: ",pos_a)
        #print("Pos b: ",pos_b)
        #print("Label a: ",label_a)
        #print("Label b: ",label_b)
        
        if(label_a!=label_b):
            if(label_a in (1,2) and label_b in (1,2)):
                continue
            else:
                return False
              
        if label_a in (3,4,5): #case: equality
            overlap_start = max(pos_a,pos_b)
            overlap_end = min(topology_a[idx+1][0],topology_b[idx+1][0])
            #print("Overlap start: ", overlap_start)
            #print("Overlap end: ",overlap_end)
            if label_a == 5:
                minimum_segment_overlap = 3
            if overlap_end-overlap_start < minimum_segment_overlap:
                return False
    return True

def train_single_epoch(model,optimizer,criterion,trainloader,type2key):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        #print(data)
        optimizer.zero_grad()
        output = model(data)
        label = label_to_tensor(data.label,type2key)
        #print("label in train: ",label)
        #print("with type: ",type(label))
        #print(output.device)
        #print(label.device)
        #print(output.shape)
        #print(torch.argmax(output,dim=1))
        #print(type(output))
        #print(label.shape)
        #print(type(label))
        #print(output)
        #print(label)
        #print(len(label))
        #print(output.shape)
        #print(output)
        #print("Output shape: ", output.shape)
        #print("Label shape: ",label.shape)
        loss = criterion(output,label) #no need to worry about batching as graph is disjoint by design -> in principle no batching
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    return running_loss/len(trainloader), output #return average epoch loss and output for the last batch  - should not be constant from start till end if learning

def quality_check_dataloaders(trainloader,valloader,testloader,mismatches,type2key):
  print("Checking lengths of label sequences vs. batch-shapes...")
  test_all_loaders = [trainloader,valloader,testloader]
  train_names = []
  val_names = []
  test_names = []
  #assure that we have no overlaps of ids, and that missing proteins are not included
  for j, loader in enumerate(test_all_loaders):
    for i, data in enumerate(loader):
    #print(i)
      if(j==0):
        train_names.append(data.id[0].replace("_ABCD",""))
      elif(j==1):
        val_names.append(data.id[0].replace("_ABCD",""))
      elif(j==2):
        test_names.append(data.id[0].replace("_ABCD",""))
      batch_tmp = data.coords.shape[0]
      label_tmp = label_to_tensor(data.label,type2key).shape[0]
      assert batch_tmp == label_tmp, f"Mismatch found for {i} {batch_tmp} {label_tmp}"

  print("... Passed test...")
  print("Testing if mismatching proteins are present in any of the sets...")
  print(train_names[0],val_names[0],test_names[0])
  assert any(x in train_names for x in mismatches)==False, "Missing proteins found in train-loader"
  assert any(x in val_names for x in mismatches)==False, "Missing proteins found in val-loader"
  assert any(x in test_names for x in mismatches)==False, "Missing proteins found in test-loader"

  print("... Passed test...")  
  print("Testing if any of the sets overlap...")
  assert len(list(set(train_names) & set(val_names)))==0, "Overlap found in train-set and val-set"
  assert len(list(set(train_names) & set(test_names)))==0, "Overlap found in train-set and test-set"
  assert len(list(set(val_names) & set(test_names)))==0, "Overlap found in val-set and test-set"
  print("... Passed test...")
  return True



