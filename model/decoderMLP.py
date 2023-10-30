import torch
import torch.nn as nn



class decoderMLP(nn.Module):
    def __init__(self, input_dim, num_layers, output_dim, dropout=0.25):
        super(decoderMLP, self).__init__()

        # Calculate the reduction factor to evenly decrease dimensionality

        
        reduction_factor = (input_dim - (output_dim * 2)) // (num_layers - 1)
        
        # Initialize a list to hold the layers of the MLP
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, input_dim - reduction_factor))
        layers.append(nn.ReLU())  # You can choose a different activation function if needed
        layers.append(nn.Dropout(dropout))
        
        # Hidden layers with residual connections
        for _ in range(num_layers - 2):  # Subtract 2 for the input and output layers
            residual = nn.Linear(input_dim - reduction_factor, input_dim - 2 * reduction_factor)
            layers.append(residual)
            layers.append(nn.ReLU())  # You can choose a different activation function if needed
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Identity())  # Identity mapping for residual connection
            input_dim -= reduction_factor
        
        # Output layer
        layers.append(nn.Linear(input_dim - reduction_factor, output_dim))
        layers.append(nn.Dropout(dropout))
        
        # Define the MLP as a Sequential model
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):

    
        return self.mlp(x)
    def mse_loss(self, pred, target):
       
        return torch.mean(torch.mean((pred - target)**2, dim=0))
    
class decoderAttention(nn.Module):
    def __init__(self,attention_dim,num_heads,num_layers_mlp,output_emb_mlp,num_layers = 2 ,dropout=0.25):
       
        super(decoderAttention, self).__init__()

        self.attnModule = nn.MultiheadAttention(attention_dim, num_heads, dropout=dropout,batch_first= True)
        
        self.mlp_shrinker = decoderMLP(attention_dim,num_layers_mlp,output_emb_mlp) 
        self.embedding_dim = attention_dim 
        
        
    def forward(self, x):

        h,_= self.attnModule(x,x,x)

        h = h.mean(axis = 1 )
        
        h = self.mlp_shrinker(h)

        
        return h

    def prepare_input(self,genre_dict, genres_list):
        # List to store genre embeddings
        genre_embeddings = []

        for genre in genres_list:
            if genre in genre_dict:
                genre_embeddings.append(genre_dict[genre])
            else:
                # If the genre is missing, pad it with zeros
                genre_embeddings.append(torch.zeros( self.embedding_dim))


        # Concatenate the genre embeddings along a new dimension (e.g., dim=1)

        input_tensor = torch.stack(genre_embeddings,dim = 0)


        return input_tensor
  
    
    

class movieTransformer(nn.Module):
    def __init__(self,attention_dim,num_heads,num_layers_mlp,output_emb_mlp,num_layers = 2 ,dropout=0.25):
       
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

    
    def prepare_input(self,genre_dict, genres_list):
        # List to store genre embeddings
        genre_embeddings = []

        for genre in genres_list:
            if genre in genre_dict:
                genre_embeddings.append(genre_dict[genre])
            else:
                # If the genre is missing, pad it with zeros
                genre_embeddings.append(torch.zeros( self.embedding_dim))


        # Concatenate the genre embeddings along a new dimension (e.g., dim=1)

        input_tensor = torch.stack(genre_embeddings,dim = 0)


        return input_tensor
  
    
    
