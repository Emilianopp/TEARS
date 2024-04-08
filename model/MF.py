import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
#import SentenceTransformer
from sentence_transformers import SentenceTransformer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Model, T5ClassificationHead,T5EncoderModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput

class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model,bias = True)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels,bias = True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class sentenceT5Classification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        
        self.classification_head = T5ClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.model_parallel = True
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        
        
        
    ) :
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,

            return_dict=return_dict
        )
        sequence_output = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)



        return Seq2SeqSequenceClassifierOutput(

            logits=logits,
            past_key_values=outputs.past_key_values,
        )
        

class sentenceT5ClassificationFrozen(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        
        self.classification_head = T5ClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()
        self.model_parallel = True
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) :
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        self.transformer.eval()
        outputs = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict
        )
        
        sequence_output = outputs[0]

        eos_mask = input_ids.eq(self.config.eos_token_id).to(sequence_output.device)

        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        batch_size, _, hidden_size = sequence_output.shape
        sentence_representation = sequence_output[eos_mask, :].view(batch_size, -1, hidden_size)[:, -1, :]
        logits = self.classification_head(sentence_representation)
        return Seq2SeqSequenceClassifierOutput(

            logits=logits,
            past_key_values=outputs.past_key_values,
        )
   
   
     

        
class MultiDAE(nn.Module):
    """
    Container module for Multi-DAE.

    Multi-DAE : Denoising Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiDAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        self.dims = self.q_dims + self.p_dims[1:]
        self.layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.dims[:-1], self.dims[1:])])

        self.drop = nn.Dropout(dropout)
        
        self.init_weights()
    
    def forward(self, input):
        h = F.normalize(input.float())
        h = self.drop(h)

        for i, layer in enumerate(self.layers):
            h = layer(h)

            h = F.tanh(h)
        return h,None,None

    def init_weights(self):
        for layer in self.layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)


class MultiVAE(nn.Module):
    """
    Container module for Multi-VAE.

    Multi-VAE : Variational Autoencoder with Multinomial Likelihood
    See Variational Autoencoders for Collaborative Filtering
    https://arxiv.org/abs/1802.05814
    """

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:
            assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
            self.q_dims = q_dims
        else:
            self.q_dims = p_dims[::-1]

        # Last dimension of q- network is for mean and variance
        temp_q_dims = self.q_dims[:-1] + [self.q_dims[-1] * 2]
        self.q_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(temp_q_dims[:-1], temp_q_dims[1:])])
        self.p_layers = nn.ModuleList([nn.Linear(d_in, d_out) for
            d_in, d_out in zip(self.p_dims[:-1], self.p_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def forward(self, input):
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def encode(self, input):
        h = F.normalize(input.float())
        h = self.drop(h)
        
        for i, layer in enumerate(self.q_layers):
            h = layer(h)
            if i != len(self.q_layers) - 1:
                h = F.tanh(h)
            else:
                mu = h[:, :self.q_dims[-1]]
                logvar = h[:, self.q_dims[-1]:]
        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    
    def decode(self, z):
        h = z
        for i, layer in enumerate(self.p_layers):
            h = layer(h)
            if i != len(self.p_layers) - 1:
                h = F.tanh(h)
        return h

    def init_weights(self):
        for layer in self.q_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.p_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0/(fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for Biases
            layer.bias.data.normal_(0.0, 0.001)

def loss_function(recon_x, x, mu, logvar, anneal=1.0):
    # BCE_fun = F.binary_cross_entropy(recon_x.float(), x.float())
    # print(f"{BCE_fun=}")
    if logvar is not None:
        BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))

        KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
    else:
        BCE = -torch.mean(torch.mean(F.log_softmax(recon_x, 1) * x, -1))
        KLD = torch.zeros_like(BCE)


    return BCE + anneal * KLD,BCE 
