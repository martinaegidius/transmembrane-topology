from typing import Optional, Set, Union

import torch
from graphein.protein.tensor.data import ProteinBatch
from torch_geometric.data import Batch
from torch_geometric.nn.models import SchNet
import torch_scatter
from proteinworkshop.types import EncoderOutput
#from torch.nn import linear
from graphein.protein.tensor.data import get_random_protein
from proteinworkshop.datasets.utils import create_example_batch


class SchNetModel(SchNet):
    #default values overwritten by cfg default settings
    def __init__(
      self,
      hidden_channels = 128, #NOTE: DEFAULTCFG 512 # Number of channels in the hidden layers
      out_dim = 32, # Output dimension of the model
      num_layers = 6, # Number of filters used in convolutional layers
      num_filters = 128, # Number of convolutional layers in the model
      num_gaussians = 50, # Number of Gaussian functions used for radial filters
      cutoff = 10.0, # Cutoff distance for interactions
      max_num_neighbors = 32, # Maximum number of neighboring atoms to consider
      readout = "add", # Global pooling method to be used
      dipole = False,
      mean =None,
      std = None,
      atomref = None
    ):
      #   hidden_channels: int = 128,#512,#128,#512,#128,
      #   out_dim: int = 32,#1,
      #   num_filters: int = 128,
      #   num_layers: int = 6,
      #   num_gaussians: int = 50,
      #   cutoff: float = 10,
      #   max_num_neighbors: int = 32,
      #   readout: str = "add",
      #   dipole: bool = False,
      #   mean: Optional[float] = None,
      #   std: Optional[float] = None,
      #   atomref: Optional[torch.Tensor] = None,
    #):
        """
        Initializes an instance of the SchNetModel class with the provided
        parameters.

        :param hidden_channels: Number of channels in the hidden layers
            (default: ``128``)
        :type hidden_channels: int
        :param out_dim: Output dimension of the model (default: ``1``)
        :type out_dim: int
        :param num_filters: Number of filters used in convolutional layers
            (default: ``128``)
        :type num_filters: int
        :param num_layers: Number of convolutional layers in the model
            (default: ``6``)
        :type num_layers: int
        :param num_gaussians: Number of Gaussian functions used for radial
            filters (default: ``50``)
        :type num_gaussians: int
        :param cutoff: Cutoff distance for interactions (default: ``10``)
        :type cutoff: float
        :param max_num_neighbors: Maximum number of neighboring atoms to
            consider (default: ``32``)
        :type max_num_neighbors: int
        :param readout: Global pooling method to be used (default: ``"add"``)
        :type readout: str
        """
        super().__init__(
            hidden_channels,
            num_filters,
            num_layers,
            num_gaussians,
            cutoff,  # None, # Interaction graph is not used
            max_num_neighbors,
            readout,
            dipole,
            mean,
            std,
            atomref,
        )
        self.readout = readout
        # Overwrite embbeding
        self.embedding = torch.nn.LazyLinear(hidden_channels)
        # Overwrite atom embedding and final predictor
        self.lin2 = torch.nn.LazyLinear(out_dim)

        #self.decoder

    @property
    def required_batch_attributes(self) -> Set[str]:
        """
        Required batch attributes for this encoder.

        - ``x``: Node features (shape: :math:`(n, d)`)
        - ``pos``: Node positions (shape: :math:`(n, 3)`)
        - ``edge_index``: Edge indices (shape: :math:`(2, e)`)
        - ``batch``: Batch indices (shape: :math:`(n,)`)

        :return: Set of required batch attributes
        :rtype: Set[str]
        """
        return {"pos", "edge_index", "x", "batch"}

    def forward(self, batch: Union[Batch, ProteinBatch]) -> EncoderOutput:
        """Implements the forward pass of the SchNet encoder.

        Returns the node embedding and graph embedding in a dictionary.

        :param batch: Batch of data to encode.
        :type batch: Union[Batch, ProteinBatch]
        :return: Dictionary of node and graph embeddings. Contains
            ``node_embedding`` and ``graph_embedding`` fields. The node
            embedding is of shape :math:`(|V|, d)` and the graph embedding is
            of shape :math:`(n, d)`, where :math:`|V|` is the number of nodes
            and :math:`n` is the number of graphs in the batch and :math:`d` is
            the dimension of the embeddings.
        :rtype: EncoderOutput
        """
        h = self.embedding(batch.x)

        u, v = batch.edge_index
        edge_weight = (batch.pos[u] - batch.pos[v]).norm(dim=-1)
        edge_attr = self.distance_expansion(edge_weight)

        for interaction in self.interactions:
            h = h + interaction(h, batch.edge_index, edge_weight, edge_attr)

        h = self.lin1(h)
        h = self.act(h)
        h = self.lin2(h)

        #add the decoder task

        return EncoderOutput(
            {
                "node_embedding": h,
                "graph_embedding": torch_scatter.scatter(
                    h, batch.batch, dim=0, reduce=self.readout
                ),
            }
        )


class DenseDecoder(torch.nn.Module):
   def __init__(self,n_classes=6,hidden_dim = 32):
      super(DenseDecoder, self).__init__()
      #self.linear1 = linear(in_features=20,out_channels=n_classes)
      self.ll = torch.nn.Linear(hidden_dim,n_classes,bias=True)
   def forward(self,x):
      #print("Decoder layer received a shape: ",x["node_embedding"].shape)
      return self.ll(x["node_embedding"])

class LSTMDecoder(torch.nn.Module):
   def __init__(self,n_classes=6, GCN_hidden_dim = 32, LSTM_hidden_dim = 32, dropout = 0.0, type="LSTMB",LSTMnormalization = False):
      super(LSTMDecoder, self).__init__()
      #self.linear1 = linear(in_features=20,out_channels=n_classes)
      if(type=="LSTMO"):
         bid = False
         self.proj = torch.nn.Linear(LSTM_hidden_dim,n_classes) #project to per class sequence
      elif(type=="LSTMB"):
         bid = True
         self.proj = torch.nn.Linear(LSTM_hidden_dim*2,n_classes) #project to per class sequence

      if(LSTMnormalization):
         self.norm = torch.nn.LayerNorm(GCN_hidden_dim)
      else:
         self.norm = None
      self.LSTM = torch.nn.LSTM(input_size = GCN_hidden_dim,hidden_size=LSTM_hidden_dim,dropout=dropout,bidirectional=bid)


      self.init_LSTM()

   def init_LSTM(self):
      # for weight in self.LSTM._all_weights:
      #    if("weight" in weight):
      #       torch.nn.init.xavier_normal_(getattr(self.LSTM,weight))
      #    if("bias" in weight):
      #       torch.nn.init.normal_(getattr(self.LSTM,weight))
      for name, param in self.LSTM.named_parameters():
         if 'bias' in name:
            torch.nn.init.constant_(param, 0.0)
         elif 'weight' in name:
            torch.nn.init.xavier_normal_(param)
      print("Initialized LSTM layers")

   def forward(self,x):
      if(self.norm!=None):
         x = self.norm(x["node_embedding"])
      else:
         x = x["node_embedding"]
      lstmoutput,_ = self.LSTM(x)
      #print("lstmoutput ", lstmoutput.shape)
      #print("state 0 shape: ",state[0].shape)
      #print("state 1 shape: ",state[1].shape)
      out = self.proj(lstmoutput)
      return out

def init_linear_weights(m):
   if isinstance(m, torch.nn.Linear):
      torch.nn.init.xavier_normal_(m.weight)
   #print("initialized linear weights")

class GraphEncDec(torch.nn.Module):
   def __init__(self,featuriser,n_classes=6,hidden_dim_GCN = 32, decoderType="linear", LSTM_hidden_dim = 0, dropout = 0.0, LSTMnormalization = False):
      #do all necessary init stuff.
      super(GraphEncDec,self).__init__()
      self.encoder = SchNetModel(out_dim=hidden_dim_GCN)
      self.featuriser = featuriser
      self.init_lazy_layers()  #initialize weights of LazyLinear layers by forwarding a random batch
      if(decoderType=="linear"):
         self.decoder =  DenseDecoder(n_classes=n_classes,hidden_dim=hidden_dim_GCN)
      elif(decoderType!="linear"): #one directional LSTM
         self.decoder = LSTMDecoder(n_classes=n_classes,GCN_hidden_dim=hidden_dim_GCN, LSTM_hidden_dim = LSTM_hidden_dim, dropout = dropout, type=decoderType,LSTMnormalization=LSTMnormalization)

      self.softmax = torch.nn.Softmax(dim=1)

   def init_lazy_layers(self):
      example_batch = create_example_batch()
      _ = self.encoder(self.featuriser(example_batch))
      print("initialized lazy layers")
      return

   def forward(self,X):
      embeddings = self.encoder(self.featuriser(X)) #returns a dict containing node_embedding and node_encoding
      predictions = self.decoder(embeddings)
      return predictions

   def predict(self,X):
      preds = self.forward(self.featuriser(X))
      sm = self.softmax(preds)
      return torch.argmax(sm,dim=1)

