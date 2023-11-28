
import torch
import json 
import os
from torch_geometric.utils import unbatch
from typing import List, Union, Dict
import numpy as np
import torchmetrics



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
    output = torch.as_tensor(output).to('cuda')
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


def saveResults(session_id,root_path,train,val,test,test_predictions,train_acc,val_acc,test_acc,train_acc_overlap,val_acc_overlap,test_acc_overlap,param_cfg):
  pwd_tmp = os.path.join(root_path,"ExperimentalResults")
  #print(pwd_tmp)
  if not (os.path.exists(pwd_tmp)):
    os.mkdir(pwd_tmp)
    print("Constructed result directory")
  
  pwd_tmp = os.path.join(pwd_tmp,session_id)
  print(pwd_tmp)
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
    
  with open(pwd_tmp+"/test_predictions.json","w") as f:
    f.write(json.dumps(test_predictions))

  with open(pwd_tmp+"/param_cfg.json","w") as f:
    f.write(json.dumps(param_cfg))

  return None


# def dataset_accuracy(model,dataloader,type2key):
#     """Calculates the per-residue topology accuracy"""
#     model.eval()
#     correct_single = 0
#     incorrect_single = 0
#     matching_overlap = 0
#     total = 0
#     with torch.no_grad():
#       for i, data in enumerate(dataloader):
#           preds = model.predict(data)
#           label = label_to_tensor(data.label,type2key) #simply concatenates across batch-dimension for single-residue evaluation
#           #single residue accuracies
#           correct_single += torch.sum((preds==label)) #no need to worry about batch
#           incorrect_single += torch.sum((preds!=label))
#           #overlap-criterion accuracies with batch-support
#           if(len(data.label)>1): #case: batch means it is a list with multiple entries -> unbatch and apply
#             unbatched = unbatch(preds,data.batch) #unbatch to tuple
#             for j, prot in enumerate(unbatched): #check sequence equality per entry
#               preds_top = label_list_to_topology(prot)
#               label_top = label_list_to_topology(label_to_tensor(data.label[j],type2key))
#               match_tmp = is_topologies_equal(preds_top,label_top)
#               matching_overlap += match_tmp #returns true if correct with at least 5 residue-overlap
#               if(match_tmp==True):
#                  print("Matched sequences. Prediction was:")
#                  print(prot)
#                  print("Label was: ")
#                  print(label_to_tensor(data.label[j],type2key))
#                  print("Encoded sequence pred: ")
#                  print(preds_top)
#                  print("Encoded sequence label: ")
#                  print(label_top)
                 
#               total += 1
#           else: #case: no batch
#             preds_top = label_list_to_topology(preds)
#             label_top = label_list_to_topology(label)
#             matching_overlap += is_topologies_equal(preds_top,label_top)
#             total += 1
      
#       overlap_accuracy = matching_overlap/total
#       #print("Total nsamples found for overlap: ",total)
#       #print("Total nsamples found for single-calculation: ",correct_single+incorrect_single)
#       single_residue_accuracy = (correct_single/(correct_single+incorrect_single)).sum().item()

#     return single_residue_accuracy, overlap_accuracy
##above is old version 

def dataset_accuracy(model,dataloader,type2key,mode="null",metrics=False): #trying with colab version
    """Calculates the per-residue topology accuracy"""
    model.eval()
    correct_single = 0
    incorrect_single = 0
    matching_overlap = 0
    total = 0
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if(metrics):
       confusion_type_labels = []
       confusion_type_preds = []
       type_predictions = []
       type_labels = []
       match_holder = []
       #probs_holder = torch.empty(1,6).to(device)
       #print("INITIALIZED PROBS_HOLDER: ",probs_holder)
       AUROC = torchmetrics.AUROC(task="multiclass",num_classes=6)
       MCROC = torchmetrics.classification.MulticlassROC(num_classes=6,thresholds=10)
    with torch.no_grad():
      sm = torch.nn.Softmax(dim=1)
      for i, data in enumerate(dataloader):
          #print(i)
          output = model(data)
          #print("output shape: ",output.shape)
          probs = sm(output)
          preds = torch.argmax(probs,dim=1)
          #preds = model.predict(data)
          label = label_to_tensor(data.label,type2key) #simply concatenates across batch-dimension for single-residue evaluation
          if(metrics):
            if(i==0):
              probs_holder = probs
              confusion_pr_labels = label
              confusion_pr_preds = preds
            else:
              probs_holder = torch.cat((probs_holder,probs),0)
              confusion_pr_labels = torch.cat((confusion_pr_labels,label),0)
              confusion_pr_preds = torch.cat((confusion_pr_preds,preds),0)
            #print("probs holder shape: ", probs_holder.shape)

          if(mode=="PRINT"):
            print("Model prediction: ",preds)
            print("Label: ",label)
          #print(preds.device)
          #print(label.device)
          #single residue accuracies
          correct_single += torch.sum((preds==label)) #no need to worry about batch
          incorrect_single += torch.sum((preds!=label))
          #overlap-criterion accuracies with batch-support
          if(len(data.label)>1): #case: batch means it is a list with multiple entries -> unbatch and apply
            # if(i==0):
            #   print("accuracy detected batch of size: ",len(data.label))
            unbatched = unbatch(preds,data.batch) #unbatch to tuple (not list)
            # if(i==0):
            #   print("before unbatching: shape ",preds.shape)
            #   print(f"After unbatching we get {len(unbatched)} samples with shapes: ...")
            for j, prot in enumerate(unbatched): #check sequence equality per entry
              # print(f"\t {prot.shape}")
              # print("The prediction is: ")
              # print(prot)
              # print("The label is: ")
              #   #print(data.label[j])#this is an issue - it is a string-tensor!
            #  print(label_to_tensor(data.label[j],type2key))
              preds_top = label_list_to_topology(prot)
              label = label_to_tensor(data.label[j],type2key)
              label_top = label_list_to_topology(label)
              match_tmp = is_topologies_equal(label_top,preds_top)
              matching_overlap += match_tmp #returns true if correct with at least 5 residue-overlap
              if(metrics):
                # #if(i==0):
                #   confusion_pr_labels = label
                #   confusion_pr_preds = prot
                # else:
                #   confusion_pr_labels = torch.cat((confusion_pr_labels,label),0)
                #   confusion_pr_preds = torch.cat((confusion_pr_preds,prot),0)
                proteinTypePred, _ = type_from_labels(prot)
                confusion_type_preds.append(proteinTypePred)
                proteinTypeLabel, _ = type_from_labels(label)
                confusion_type_labels.append(proteinTypeLabel)
                match_holder.append(match_tmp)
                type_predictions.append(proteinTypePred)
                type_labels.append(proteinTypeLabel)
              total += 1
          else: #case: no batch
            preds_top = label_list_to_topology(preds)
            label_top = label_list_to_topology(label)
            match_tmp = is_topologies_equal(label_top,preds_top)
            matching_overlap += match_tmp
            if(metrics):
                # if(i==0):
                #   confusion_pr_labels = label
                #   confusion_pr_preds = preds
                # else:
                #   confusion_pr_labels = torch.cat((confusion_pr_labels,label),0)
                #   confusion_pr_preds = torch.cat((confusion_pr_preds,preds),0)
                proteinTypePred, _ = type_from_labels(preds)
                confusion_type_preds.append(proteinTypePred)
                proteinTypeLabel, _ = type_from_labels(label)
                confusion_type_labels.append(proteinTypeLabel)
                match_holder.append(match_tmp)
                type_predictions.append(proteinTypePred)
                type_labels.append(proteinTypeLabel)
                #print("residue labels shape:", confusion_pr_labels.shape)
                
                  

            if(mode=="PRINT"):
              print(f"Testing overlap for protein {i}: ")
              print("Preds_top: ",preds_top)
              print("label_top: ",label_top)
              print("Matching result: ",match_tmp)
            total += 1

      overlap_accuracy = matching_overlap/total
      print("Total nsamples found for overlap: ",total)
      print("Total nsamples found for single-calculation (should equal batchsize): ",correct_single+incorrect_single)
      single_residue_accuracy = (correct_single/(correct_single+incorrect_single)).item()
    if(metrics):
       confusion_pr_labels = confusion_pr_labels.to(torch.int32)
       confusion_type_pred_T = torch.Tensor(confusion_type_preds)
       confusion_type_label_T = torch.Tensor(confusion_type_labels) 
       confmatTop = torchmetrics.ConfusionMatrix(task="multiclass",num_classes=6).to(device)
       try:
          confmatTop_ret = confmatTop(confusion_pr_preds,confusion_pr_labels)
           
       except: 
          confmatTop_ret = "NULL"
          print("Couldnt make per residue confmat due to error!. Printing labels and preds for per-residue: ")
          print(confusion_pr_labels)
          print(confusion_pr_preds)

          print("The unique entries in each are: ")
          print("Predictions: ",torch.unique(confusion_pr_preds))
          print("Labels: ",torch.unique(confusion_pr_labels))

       try:
          confmatType = torchmetrics.ConfusionMatrix(task="multiclass",num_classes=6) #is on cpu 
          confmatType_ret = confmatType(confusion_type_pred_T,confusion_type_label_T)
          
       except: 
          print("Couldnt make type confmat due to error!. Printing labels and preds for type: ")
          print("prediction tensor")
          print(confusion_type_pred_T)
          print(confusion_type_pred_T.shape)
          print("label tensor")
          print(confusion_type_label_T)
          print(confusion_type_label_T.shape)
          
          print("The unique entries in each are: ")
          print("Predictions: ",torch.unique(confusion_type_pred_T))
          print("Labels: ",torch.unique(confusion_type_label_T))
          
          confmatType_ret = "NULL"
          
          
       try:   
          AUROC_ret = AUROC(probs_holder,confusion_pr_labels.squeeze())
       except: 
          AUROC_ret = "NULL"
          print("Issue with AUROC")
       try: 
          probs_holder_d = probs_holder.detach().cpu()
          confusion_pr_labels_d = confusion_pr_labels.squeeze().detach().cpu().long()
          #print(probs_holder_d.shape)
          #print(confusion_pr_labels_d.shape)
          #print(probs_holder_d)
          #print(confusion_pr_labels_d)
          MCROC.update(probs_holder_d,confusion_pr_labels_d) #update object for plotting
       except:
          MROC = "NULL"
          print("ISSUES WITH ROC METRICS")
          print("Probs holder shape: ",probs_holder_d.shape)
          print("Shape of labels after squeezing: ",confusion_pr_labels_d.shape)
          print(probs_holder_d)
          print(confusion_pr_labels_d)
          print("SUM: ")
          print(torch.sum(probs_holder_d,dim=1))

       print("----------- PERFORMANCE EVALUATION USING THEIR METRICS --------------")
       top_confusion_matrix = torch.zeros(size=(6,6)).long() #last column is for counting correct topologies for the given class
       for actual_type,predicted_type,prediction_topology_match in zip(type_labels,type_predictions,match_holder):
         #print(actual_type,predicted_type,prediction_topology_match)
         if actual_type == predicted_type:
                # if we guessed the type right for SP+GLOB or GLOB,
                # count the topology as correct
           if actual_type == 2 or actual_type == 3 or prediction_topology_match:
             top_confusion_matrix[actual_type][5] += 1
           else:
             top_confusion_matrix[actual_type][predicted_type] += 1

         else:
            top_confusion_matrix[actual_type][predicted_type] += 1
       #print(top_confusion_matrix)

       result_metrics = topology_accuracies(top_confusion_matrix)
        
       return single_residue_accuracy, overlap_accuracy,confmatTop_ret,confmatType_ret,AUROC_ret,MCROC,result_metrics


    else:
      return single_residue_accuracy, overlap_accuracy


def calculate_acc(correct, total):
    total = total.float()
    correct = correct.float()
    if total == 0.0:
        return 1
    return correct / total


def topology_accuracies(confusion_matrix):
    type_correct_ratio = \
    calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()) + \
    calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()) + \
    calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()) + \
    calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()) + \
    calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum())
    type_accuracy = float((type_correct_ratio / 5))

    tm_accuracy = float(calculate_acc(confusion_matrix[0][5], confusion_matrix[0].sum()))
    sptm_accuracy = float(calculate_acc(confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_accuracy = float(calculate_acc(confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_accuracy = float(calculate_acc(confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_accuracy = float(calculate_acc(confusion_matrix[4][5], confusion_matrix[4].sum()))
    
    tm_type_acc = float(calculate_acc(confusion_matrix[0][0] + confusion_matrix[0][5], confusion_matrix[0].sum()))
    tm_sp_type_acc = float(calculate_acc(confusion_matrix[1][1] + confusion_matrix[1][5], confusion_matrix[1].sum()))
    sp_type_acc = float(calculate_acc(confusion_matrix[2][2] + confusion_matrix[2][5], confusion_matrix[2].sum()))
    glob_type_acc = float(calculate_acc(confusion_matrix[3][3] + confusion_matrix[3][5], confusion_matrix[3].sum()))
    beta_type_acc = float(calculate_acc(confusion_matrix[4][4] + confusion_matrix[4][5], confusion_matrix[4].sum()))

    print("--------------- Topology accuracy results: -----------------------")
    print("\t tm accuracy: ",tm_accuracy)
    print("\t sptm accuracy: ",sptm_accuracy)
    print("\t sp accuracy: ",sp_accuracy)
    print("\t glob accuracy: ",glob_accuracy)
    print("\t beta accuracy: ",beta_accuracy)
    
    print("--------------- Type accuracy results: ------------------")
    print("\t type accuracy: ", type_accuracy)
    print("\t tm type accuracy: ",tm_type_acc)
    print("\t tm sp type accuracy: ",tm_sp_type_acc)
    print("\t sp type accuracy: ",sp_type_acc)
    print("\t glob type accuracy: ",glob_type_acc)
    print("\t beta type accuracy: ",beta_type_acc)

    result_dict = {"tm acc":tm_accuracy,"sptm acc":sptm_accuracy,"sp acc":sp_accuracy,"glob acc":glob_accuracy,"beta acc":beta_accuracy,"type acc":type_accuracy,
                   "tm type acc":tm_type_acc,"tmsp type acc":tm_sp_type_acc,"sp type acc":sp_type_acc,"glob type acc":glob_type_acc,"beta type acc":beta_type_acc}
    
    return result_dict


    
    



def type_from_labels(label):
    """
    Function that determines the protein type from labels

    Dimension of each label:
    (len_of_longenst_protein_in_batch)

    # Residue class
    0 = inside cell/cytosol (I)
    1 = Outside cell/lumen of ER/Golgi/lysosomes (O)
    2 = periplasm (P)
    3 = signal peptide (S)
    4 = alpha membrane (M)
    5 = beta membrane (B)
    
    B (contains) -> beta
    I (only) -> globular
    Both S and M (contains) -> SP + alpha(TM)
    M (contains) -> alpha(TM)
    S (only) -> signal peptide
    P (contains) -> beta membrane

    # Protein type class
    0 = TM
    1 = SP + TM
    2 = SP
    3 = GLOBULAR
    4 = BETA
    """
    #print("RECEIVED LABEL: ",label)
    if 5 in label: #case: BETA-barrel (fixed from theirs)
        ptype = 4

    elif all(element == 0 for element in label): #case: globular
        ptype = 3

    elif 3 in label and 4 in label: #SP+TM
        ptype = 1

    elif 3 in label: #Signal peptide
       ptype = 2

    elif 4 in label: #Alpha TM 
        ptype = 0

    elif all(x == 0 or x == -1 for x in label): #case: globular
        ptype = 3

    else: #case: invalid
        ptype = 5

    conversionDict = {"0":"TM","1":"SP+TM","2":"SP","3":"GLOBULAR","4":"BETA","5":"INVALID"}
    return ptype, conversionDict[str(ptype)]

def log_batch_elementwise_accuracies(unbatched_preds,data,type2key):
  #data is batched format, predictions are unbatched.

  
  ids = []
  labels = []
  preds = []
  label_type = []
  pred_type = []
  overlap_match = []
  accuracy = []
  if(isinstance(unbatched_preds,list)):#CASE: BATCH
     num_samples = len(unbatched_preds)
  else:
     unbatched_preds = [unbatched_preds]
     num_samples = 1
  for j in range(num_samples):
    #print(f"Investigating samle no ", j)
    prot = unbatched_preds[j] #numeric prediction tensor
    label = label_to_tensor(data.label[j],type2key) #label is a string, label to tensor gives numeric encoding of label string
    #print("FOUND PREDICTION: ",prot)
    #print("FOUND LABEL: ",label)
    _, protein_type_label = type_from_labels(label)
    _, protein_type_prediction = type_from_labels(prot)
    #print("prot: ",prot)
    #print("Label: ", label)
    #print("Id: ", data.id[j])
    
    n_correct = torch.sum((prot==label)) #no need to worry about batch
    n_incorrect = torch.sum((prot!=label))
    preds_top = label_list_to_topology(prot)
    label_top = label_list_to_topology(label)
    match_tmp = is_topologies_equal(label_top,preds_top)
    overlap_match.append(match_tmp)
    ids.append(data.id[j].replace("_ABCD",""))
    labels.append(topology_list_to_string(label_top))
    preds.append(topology_list_to_string(preds_top))
    accuracy.append((n_correct/(n_correct+n_incorrect)).item())
    label_type.append(protein_type_label)
    pred_type.append(protein_type_prediction)

  # print(ids)
  # print(labels)
  # print(preds)
  # print(accuracy)
  # print(overlap_match)
  # print(label_type)
  # print(pred_type)
  return ids, labels, preds, accuracy, overlap_match,label_type,pred_type
    

#dataset_accuracy(model,trainloader,type2key)
#dataset_accuracy(model,testloader,type2key)

def topology_list_to_string(top):
  str_tmp = "["
  for idx, entry in enumerate(top):
    pos = int(entry[0].item())
    label = int(entry[1].item())
    str_tmp += "("+str(pos) + "," + str(label) + ")"

    if(0<idx<len(top)-1):
        str_tmp += ","
    
  str_tmp += "]"
  return str_tmp

def is_topologies_equal(topology_a, topology_b, minimum_seqment_overlap=5):
    """
    Checks whether two topologies are equal.
    E.g. [(0,A),(3, B)(7,C)]  is the same as [(0,A),(4, B)(7,C)]
    But not the same as [(0,A),(3, C)(7,B)]

    Parameters
    ----------
    topology_a : list of torch.Tensor
        First topology. See label_list_to_topology.
    topology_b : list of torch.Tensor
        Second topology. See label_list_to_topology.
    minimum_seqment_overlap : int
        Minimum overlap between two segments to be considered equal.

    Returns
    -------
    bool
        True if topologies are equal, False otherwise.
    """

    if isinstance(topology_a[0], torch.Tensor):
        topology_a = list([a.cpu().numpy() for a in topology_a])
    if isinstance(topology_b[0], torch.Tensor):
        topology_b = list([b.cpu().numpy() for b in topology_b])

    #print("Label topology: ",topology_a)
    #print("Prediction topology: ",topology_b)
    #print("Label topology length: ",len(topology_a))
    #print("Prediction topology length: ",len(topology_b))
    
    if len(topology_a) != len(topology_b): #note: this will return false if the number of topology-switches are incorrect! 
        return False
    for idx, (_position_a, label_a) in enumerate(topology_a):
        if label_a != topology_b[idx][1]:
            if (label_a in (1,2) and topology_b[idx][1] in (1,2)): # assume O == P, ie. we accept if these are interchanged
                continue
            else:
                return False #other topologies: do not accept
        if label_a in (3, 4, 5):
            if label_a == 5:
                # Set minimum segment overlap to 3 for Beta regions
                minimum_seqment_overlap = 3

            if(idx<len(topology_a)-1): #don't want to go beyond the last idx, this raises error
              overlap_segment_start = max(topology_a[idx][0], topology_b[idx][0])
              overlap_segment_end = min(topology_a[idx + 1][0], topology_b[idx + 1][0])
              if overlap_segment_end - overlap_segment_start < minimum_seqment_overlap:
                  return False
            else: #only checking last IDX overlap
               nonOverlapping = max(topology_a[idx][0],topology_b[idx][0])-min(topology_a[idx][0],topology_b[idx][0])
               if nonOverlapping > minimum_seqment_overlap:
                  return False 
    return True

def label_list_to_topology(labels: Union[list[int], torch.Tensor]) -> list[torch.Tensor]:
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
      #print("Detected list")
      labels = torch.LongTensor(labels)

    if isinstance(labels, torch.Tensor):
      #print("Detected tensor")
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


    

# def sequence_equality(topology_a,topology_b,minimum_segment_overlap=5):
#     if isinstance(topology_a[0], torch.Tensor):
#             #print("Converted a to list!")
#             topology_a = list([a.cpu().numpy() for a in topology_a])
#     if isinstance(topology_b[0], torch.Tensor):
#             #print("Converted b to list!")
#             topology_b = list([b.cpu().numpy() for b in topology_b])
#     if len(topology_a) != len(topology_b):
#             return False
    
#     for idx, (pos_a,label_a) in enumerate(topology_a):
#         #print("checking: ",idx)
#         pos_b, label_b = topology_b[idx]
#         #print("Pos a: ",pos_a)
#         #print("Pos b: ",pos_b)
#         #print("Label a: ",label_a)
#         #print("Label b: ",label_b)
        
#         if(label_a!=label_b):
#             if(label_a in (1,2) and label_b in (1,2)):
#                 continue
#             else:
#                 return False
              
#         if label_a in (3,4,5): #case: equality
#             overlap_start = max(pos_a,pos_b)
#             overlap_end = min(topology_a[idx+1][0],topology_b[idx+1][0])
#             #print("Overlap start: ", overlap_start)
#             #print("Overlap end: ",overlap_end)
#             if label_a == 5:
#                 minimum_segment_overlap = 3
#             if overlap_end-overlap_start < minimum_segment_overlap:
#                 return False
#     return True

def train_single_epoch(model,optimizer,criterion,trainloader,type2key,gradclip=False,clipval=None):
    running_loss = 0.0
    model.train()
    for i, data in enumerate(trainloader):
        #print(data)
        output = model(data)
        label = label_to_tensor(data.label,type2key)
        loss = criterion(output,label) #no need to worry about batching as graph is disjoint by design -> in principle no batching
        optimizer.zero_grad()           
        loss.backward()
        if(gradclip):
           torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm = clipval, norm_type=2.0) #using a value of 10 
        optimizer.step()
        running_loss += loss.item()

    return running_loss, output #return average epoch loss and output for the last batch  - should not be constant from start till end if learning



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



