import graphein.protein.tensor as gpt
from torch_geometric.data import Dataset as gDataset
import glob
import os.path as osp
import torch
from graphein.protein.utils import download_alphafold_structure
import os

class transmembraneDataset(gDataset):
    """root: where the dataset should be stored"""
    def __init__(self, root, setType, proteinlist, mismatching_proteins,labelDict, flush_files = False, transform=None, pre_transform=None,pre_filter=None):
        self.proteinlist = proteinlist
        self.mismatches = mismatching_proteins
        self.proteinlist = [x for x in proteinlist if x not in self.mismatches]
        self.conversionDict = {} #save data-conversion table between saved pt files and proteinnames
        self.root_local = root
        self.setType = setType
        self.labels = labelDict
        self.flush_files = flush_files
        #print("self root local: ",self.root_local)
        #print("protein list: ")
        #print(self.proteinlist)
        #print("pdb dir: ",self.pdb_dir)
        super().__init__(root, transform, pre_transform, pre_filter)


    @property ##source directory of files - this is pdb-files in this case
    def raw_file_names(self): #returns a list of all paths valid for entries in proteinlist
        #full_file_list = glob.glob(self.pdb_dir+"/*.pdb") #get all downloaded paths
        full_file_list = glob.glob(self.root_local+"raw/*.pdb") #get all downloaded paths

        #find matches so it matches the split
        res = list(
            set([sub1 for ele1 in full_file_list for sub1 in self.proteinlist if sub1 in ele1]))
        self.proteinlist = res #NOTE: OVERWRITES VALID FILES
        
        #res = [self.pdb_dir + x +".pdb" for x in res]
        #print(res)
        res = [x +".pdb" for x in res]
        
        #print("RESULTING RAW FILENAMES")
        #print(res)

        return res

    @property
    def processed_file_names(self):
        return [self.setType+"_"+"protein_ "+ str(i) + ".pt" for i,_ in enumerate(self.proteinlist)]

    def download(self):
        for protein_name in self.proteinlist:
            _ = download_alphafold_structure(protein_name, version=4,out_dir = self.root_local+"raw/", aligned_score=True)
        return

    def process(self):
        idx = 0
        #print("raw paths: ")
        #for raw_path in self.raw_paths: #this one adds a /raw/ for some reason
         #   print(raw_path)

        #print("Processed dir: ",self.processed_dir)
        all_files_tmp = glob.glob(osp.join(self.processed_dir, f'{self.setType}_protein_*.pt'))
        if(len(all_files_tmp)>=len(self.proteinlist) and self.flush_files==False): ##case: all proteins are present
            #print(f"Length of proteinnames {len(self.proteinlist)} is == or larger than length of existing proteins in processed dir {len(all_files_tmp)}")
            return

        for i, raw_path in enumerate(self.raw_paths):
            print(f"running protein {i}/{len(self.raw_paths)} in set: {self.setType}")
            tmp_file_check = osp.join(self.processed_dir, f'{self.setType}_protein_{idx}.pt')
            if(os.path.isfile(tmp_file_check) and self.flush_files==False): #case: some files may already exist due to interruption of earlier command
                pass
            else:
                data = gpt.io.protein_to_pyg(path=raw_path,
                    chain_selection=["A", "B", "C", "D"], # Select all 4 chains
                    deprotonate=True, # Deprotonate the structure
                    keep_insertions=False, # Remove insertions
                    keep_hets=[], # Remove HETATMs
                    model_index=1) # Select the first model
                    # Can select a subset of atoms with atom_types=...

                #print(data)
                id = data["id"].replace("_ABCD","")#remove chain selection in naming for saving to dictionary
                if(id in self.mismatches):
                  print("ERROR - DIDN'T EXCLUDE SAMPLE!")
                  print(id)
                  print(i)
                  break
                #print("working on protein: f{id}")
                protein = gpt.Protein().from_data(data) #load protein from pdb-data
                #add edges etc
                protein.batch = protein.coords
                protein.edges("knn_8",cache="edge_index")
                protein.pos = protein.coords[:,1,:]
                protein.x = protein.residue_type
                #print(protein)
                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    protein = self.pre_transform(protein)

                protein.label = self.labels[id] #fetch labelled sequence for the given protein and assign to graph-field

                protein = protein.to_data() #returns a torch geometric data object used for batching https://github.com/a-r-j/graphein/blob/master/graphein/protein/tensor/data.py#L266
                #save conversion for later reference in case all goes south
                self.conversionDict[id] = f'{self.setType}_protein_{idx}.pt'
                #print("the path join has arg:")
                #print(osp.join(self.processed_dir, f'protein_{idx}.pt'))
                torch.save(protein, osp.join(self.processed_dir, f'{self.setType}_protein_{idx}.pt'))
            idx += 1
            #if(idx==10): #for debugging
            #    return

    def len(self):
        return len(self.processed_file_names)

    def get(self,idx):
        data = torch.load(osp.join(self.processed_dir, f'{self.setType}_protein_{idx}.pt'),map_location='cuda')
        #need to also return labels for the given protein
        return data

    def get_protein_label_dict(self):
        return self.labels
