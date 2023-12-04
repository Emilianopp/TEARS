import torch
import torch.nn as nn
from model.decoderMLP import decoderMLP
class movieTransformer(nn.Module):
    def __init__(self,attention_dim,num_heads,num_layers_mlp,output_emb_mlp,num_layers = 2 ,dropout=0.25,bias = False):
       
        super(movieTransformer, self).__init__()
        
        #make a loop to make transformer layers

        self.nn_list= nn.ModuleList()
        

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=attention_dim, nhead=num_heads, dim_feedforward=attention_dim, dropout=dropout, activation='relu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.mlp_shrinker = decoderMLP(attention_dim,num_layers_mlp,output_emb_mlp) 
        self.embedding_dim = attention_dim
        
    def forward(self, x):

        h = self.transformer_encoder(x)
        h = h.mean(axis = 1 )
        h = self.mlp_shrinker(h)

        
        return h