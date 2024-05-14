from helper.eval_metrics import Recall_at_k_batch,NDCG_binary_at_k_batch,MRR_at_k
from torch.distributions import Normal, kl_divergence, RelaxedOneHotCategorical

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
#import SentenceTransformer
from transformers.models.phi import PhiPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from sentence_transformers import SentenceTransformer
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5Model, T5ClassificationHead,T5EncoderModel
from transformers.models.phi.modeling_phi import PhiConfig,PhiModel
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import Seq2SeqSequenceClassifierOutput
from transformers import PreTrainedModel
from transformers import AutoConfig
from transformers import AutoModel,BitsAndBytesConfig
from peft import LoraConfig, TaskType, PeftModel,get_peft_model,prepare_model_for_kbit_training
from transformers import T5Tokenizer ,AutoTokenizer
import ot 
#import deepbopy
from copy import deepcopy



def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
        x = F.dropout(x, p=dropout_rate, training=self.training)
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    


class RecVAE(nn.Module):
    def __init__(self, p_dims,dropout,gamma):
        super(RecVAE, self).__init__()
        hidden_dim = p_dims[0]
        latent_dim = p_dims[0]
        input_dim = p_dims[-1]
        self.dropout = dropout
        self.gamma = gamma 

        self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def encode(self,user_ratings):
        mu, logvar = self.encoder(user_ratings, dropout_rate=self.dropout)    
        
        return mu,logvar
    def decode(self,z):
        return self.decoder(z)
    def forward(self, user_ratings,all_ratings, beta=None, gamma=1, dropout_rate=0.5, calculate_loss=False):
        # print(f"{user_ratings.sum()=}")
        mu, logvar = self.encoder(user_ratings, dropout_rate=self.dropout)    



        z = self.reparameterize(mu, logvar)

        x_pred = self.decoder(z)
        
        if calculate_loss:
            if self.gamma:
                norm = user_ratings.sum(dim=-1)
                kl_weight = self.gamma * norm
            elif beta:
                kl_weight = beta

            mll = (F.log_softmax(x_pred, dim=-1) * all_ratings).sum(dim=-1).mean()

            kld = (log_norm_pdf(z, mu, logvar) - self.prior(all_ratings, z)).sum(dim=-1).mul(kl_weight).mean()

            negative_elbo = -(mll - kld)
            
            return x_pred, negative_elbo
            
        else:
            return x_pred,mu,logvar,z
    def update_prior(self):
        self.prior.encoder_old.load_state_dict(deepcopy(self.encoder.state_dict()))



# class RecVAE(nn.Module):
#     def __init__(self, p_dims,dropout = .5,gamma = .005):
#         super(RecVAE, self).__init__()
#         hidden_dim=latent_dim = p_dims[0]
#         input_dim = p_dims[1]
#         self.gamma = gamma
#         self.dropout = dropout
#         self.q = Encoder(hidden_dim, latent_dim, input_dim)
#         self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
#         self.decoder = nn.Linear(latent_dim, input_dim)
#         # self.__init__weights(fan_in = latent_dim,fan_out = input_dim)
        
#     def __init__weights(self, fan_in, fan_out):
#         std= np.sqrt(2.0 / (fan_in + fan_out))
#         self.decoder.weight.data.normal_(0.0, std)
         
    
#     def reparameterize(self, mu, logvar):
#         if self.training:
#             std = torch.exp(0.5*logvar)
#             eps = torch.randn_like(std)
#             return eps.mul(std).add_(mu)
#         else:
#             return mu
#     def encode(self,user_ratings):
#         mu, logvar = self.q(user_ratings, dropout_rate=self.dropout)    
        
#         return mu,logvar
#     def decode(self,z):
#         return self.decoder(z)
#     def forward(self, user_ratings, beta=None,  calculate_loss=False):
#         mu, logvar = self.q(user_ratings, dropout_rate=self.dropout)    
#         z = self.reparameterize(mu, logvar)
#         x_pred = self.decoder(z)
        
#         if calculate_loss:
#             if self.gamma:
#                 norm = user_ratings.sum(dim=-1)
#                 kl_weight = self.gamma * norm
#             elif beta:
#                 kl_weight = beta

#             mll = (F.log_softmax(x_pred, dim=-1) * user_ratings).sum(dim=-1).mean()
#             kld = (log_norm_pdf(z, mu, logvar) - self.prior(user_ratings, z)).sum(dim=-1).mul(kl_weight).mean()
#             negative_elbo = -(mll - kld)
            
#             return x_pred, negative_elbo
            
#         else:

#             return x_pred,mu,logvar,z

#     def update_prior(self):
#         self.prior.encoder_old.load_state_dict(deepcopy(self.q.state_dict()))
        


class T5ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.d_model, config.d_model,bias = False)
        self.dropout = nn.Dropout(p=config.classifier_dropout)
        self.out_proj = nn.Linear(config.d_model, config.num_labels,bias = False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class FiLMLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(FiLMLayer, self).__init__()

        self.gamma_layer = nn.Linear(int(input_size), int(output_size))
        self.beta_layer = nn.Linear(int(input_size), int(output_size))

        self.__init__weights(fan_in = input_size,fan_out = output_size)
       
    def forward(self, x):
        gamma = self.gamma_layer(x)

        beta = self.beta_layer(x)
        return gamma, beta

    def __init__weights(self, fan_in, fan_out):
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.gamma_layer.weight.data.normal_(0.0, std)
        self.gamma_layer.bias.data.normal_(0.0, 0.001)
        self.beta_layer.weight.data.normal_(0.0, std)
        self.beta_layer.bias.data.normal_(0.0, 0.001)

class VaeClassifier(nn.Module):
    def __init__(self, config,p_dims,dropout):
        super(VaeClassifier, self).__init__()

        self.llm_dim = config.d_model
        self.film_layer = FiLMLayer(config.d_model, 200 * 2)  # Adjust dimensions as needed
        self.dropout = nn.Dropout(p=dropout)
        self.VAE = MultiVAE(p_dims, dropout = dropout)

    def forward(self, data_tensor, hidden_states):
        # Generate gamma and beta from hidden_states
        hidden_states = self.dropout(hidden_states)
        gamma, beta = self.film_layer(hidden_states)
        # print(f"{gamma.sum()=}")
        gamma_mu = gamma[:, :200]
        gamma_logvar = gamma[:, 200:]
        # print(f"{gamma_logvar.sum()=}")
        beta_mu = beta[:, :200]
        beta_logvar = beta[:, 200:]
        # Apply FiLM transformation to data_tensor

        mu, logvar = self.VAE.encode(data_tensor)

        

        film_mu = gamma_mu * mu + beta_mu

        film_var = gamma_logvar * logvar + beta_logvar
        
        # modulated_data_tensor = F.gelu(modulated_data_tensor)

        z = self.VAE.reparameterize(film_mu, film_var)
        # Decode z
        return self.VAE.decode(z),mu,logvar



class sentenceT5ClassificationVAE(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        p_dims = [200,config.num_labels]
        # self.linear = nn.Linear(config.d_model,config.d_model//2,bias = False)
        self.classifier = VaeClassifier(config,p_dims,dropout=config.classifier_dropout)

        self.model_parallel = True
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
    ) :
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        logits,mu,logvar = self.classifier_forward(data_tensor,sentence_rep)
        
        
        return logits,mu,logvar
    
  
    def classifier_forward(self,data_tensor,hidden_states):
        # hidden_states = self.linear(hidden_states)
        # hidden_states = F.gelu(hidden_states)
        logits = self.classifier(data_tensor,hidden_states)
        return logits
    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
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
        return sentence_representation
    
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ):
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        semtemce_rep = self.llm_forward(**tokenized_summary)

        logits = self.classifier_forward(data_tensor.unsqueeze(0),semtemce_rep)[0]
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()


class otVAE (T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config,llm,vae,prior,epsilon):
        super().__init__(config)
        self.transformer =   llm
        p_dims = [200,config.num_labels]
        self.epsilon = epsilon
       
        self.vae = vae
        # self.joint_decoder = nn.Linear(400,config.num_labels,bias = None )
        # self.__init__weights()
        self.model_parallel = True
        self.prior = prior

            
    # def __init__weights(self):


    #         fan_in = self.joint_decoder.weight.size()[1]
    #         fan_out = self.joint_decoder.weight.size()[0]
    #         std = np.sqrt(2.0 / (fan_in + fan_out))
    #         self.joint_decoder.weight.data.normal_(0.0, std)

            
        
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        alpha = .5
    ) :

        vae = self.vae
        hidden_states = self.transformer.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        mu,logvar = self.transformer.encode(hidden_states)
        z_text = vae.reparameterize(mu, logvar)
        
        prior_mu, prior_logvar = vae.encode(data_tensor)
        z_rec = vae.reparameterize(prior_mu, prior_logvar)

        z_merged = (1-alpha)*z_text + alpha*z_rec 


        logits_merged = vae.decode(z_merged)

        logits_rec = vae.decode(z_rec)
        logits_text = vae.decode(z_text)

        
        
        return logits_merged,logits_rec,logits_text,mu,logvar,prior_mu,prior_logvar,z_rec,None
    
    
    
    def classifier_forward(self,data_tensor,hidden_states):
        # hidden_states = self.linear(hidden_states)
        # hidden_states = F.gelu(hidden_states)
        logits = self.classifier(data_tensor,hidden_states)
        return logits


    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
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
        return sentence_representation
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,alpha = .5):
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        logits = self.forward(data_tensor = data_tensor,input_ids = tokenized_summary['input_ids'],attention_mask = tokenized_summary['attention_mask'],alpha = alpha)[0]
        #check shape of data_tensor 
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()



        
    

class T5Vae(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config,prior,epsilon,concat=False):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        p_dims = [200,config.num_labels]
        self.epsilon = epsilon
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 800, bias=False),
        )
        self.vae = prior
        self.model_parallel = True
        self.concat = concat
        if concat: 
            self.concat_mlp = nn.Linear(800,400,bias = False)
        self.__init__weights()

            
    def __init__weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size()[1]
                fan_out = layer.weight.size()[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
        if self.concat:
            fan_in = self.concat_mlp.weight.size()[1]
            fan_out = self.concat_mlp.weight.size()[0]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            self.concat_mlp.weight.data.normal_(0.0, std)
            
    def set_vae(self,vae):
        self.vae = vae 
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        alpha = .5,
        neg = False
    ) :


        # print(f"{self.vae.modules_to_save['default']=}")
        try:
            vae = self.vae.modules_to_save['default']
        except:
            vae = self.vae
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:400],sentence_rep[:,400:]
        z_text = self.reparameterize(mu, logvar)
        prior_mu, prior_logvar = vae.encode(data_tensor)
        z_rec = vae.reparameterize(prior_mu, prior_logvar)

        #map z_rec onto the text 
        if self.concat:
            z_merged = torch.cat([z_text,z_rec],dim = 1)
            z_merged = self.concat_mlp(z_merged) if alpha != 0 else z_text 
        elif neg: 

            z_merged =  + (z_rec -z_text)/2
            
        else:

            z_merged = (1-alpha)*z_text + alpha*z_rec 


        logits_merged = vae.decode(z_merged)
        logits_rec = vae.decode(z_rec)
        logits_text = vae.decode(z_text)
        

        
        
        return logits_merged,logits_rec,logits_text,mu,logvar,prior_mu,prior_logvar,z_rec,None
    
    
    def classifier_forward(self,data_tensor,hidden_states,alpha = None ):
        # hidden_states = self.linear(hidden_states)
        # hidden_states = F.gelu(hidden_states)
        logits = self.classifier(data_tensor,hidden_states)
        return logits


    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
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
        return sentence_representation
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,alpha = .5,neg = False):
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        logits = self.forward(data_tensor = data_tensor,input_ids = tokenized_summary['input_ids'],attention_mask = tokenized_summary['attention_mask'],alpha = alpha,neg = neg)[0]
        #check shape of data_tensor 
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()



class sentenceT5ClassificationPooling(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)

        # self.linear = nn.Linear(config.d_model,config.d_model//2,bias = False)
        p_dims = [200,config.num_labels]

        self.VAE = MultiVAE(p_dims, dropout = config.classifier_dropout)
        #make a two layer mlp that goes from config.module_dim to 200 

        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_model//2,bias = False),
            nn.Tanh(),
            nn.Linear(config.d_model//2, 400,bias = False)
        )
        
        
        self.__init__weights()
        self.model_parallel = True
    def __init__weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                fan_in, fan_out = module.weight.size()
                std = np.sqrt(2.0 / (fan_in + fan_out))
                module.weight.data.normal_(0, std)
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        gamma = .5,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
    ) :
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        sentence_rep = self.mlp(sentence_rep)
        sent_mu,sent_logvar = sentence_rep[:,:200],sentence_rep[:,200:]
        


        vae_mu,vae_logvar = self.VAE.original_module.encode(data_tensor)
        

        mu = (gamma * sent_mu) + ((1-gamma) * vae_mu)
        


        logvar = (gamma * sent_logvar) + ((1-gamma) * vae_logvar)
        
        z = self.VAE.original_module.reparameterize(mu, logvar)
        
        logits = self.VAE.original_module.decode(z)
        
        return logits,mu,logvar
    def classifier_forward(self,data_tensor,hidden_states):
        # hidden_states = self.linear(hidden_states)
        # hidden_states = F.gelu(hidden_states)
        logits = self.classifier(data_tensor,hidden_states)
        return logits
    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
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
        return sentence_representation
    
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,neg = False):
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        semtemce_rep = self.llm_forward(**tokenized_summary)

        logits = self.classifier_forward(data_tensor.unsqueeze(0),semtemce_rep)[0]
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()


        



class sentenceT5Classification(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        
        self.classifier = T5ClassificationHead(config)
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
        labels_tr: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        data_tensor = None
    ) :
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        logits = self.classifier_forward(sentence_rep)
        
        return logits,None,None
    def classifier_forward(self,hidden_states):
        logits = self.classifier(hidden_states)
        return logits
    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                ):

        
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
        return sentence_representation
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,**kwargs):
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        semtemce_rep = self.llm_forward(**tokenized_summary)

        logits = self.classifier_forward(semtemce_rep)
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 

        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()
        

class sentenceT5Vae(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)

        self.classifier = MultiVAE(q_dims = [config.d_model,800,400],p_dims = [400,config.num_labels],dropout = config.classifier_dropout)
     

        # self.post_init()
        self.model_parallel = True
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        labels_tr: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        data_tensor = None
    ) :
        
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)

        logits,mu,logvar = self.classifier_forward(sentence_rep)

        
        return logits,mu,logvar
        
    def classifier_forward(self,hidden_states):
        logits,mu,logvar = self.classifier(hidden_states)
        return logits,mu,logvar
    def encode(self,data_tensor):


        return self.classifier.original_module.encode(data_tensor)
    def decode (self,z):
        return self.classifier.original_module.decode(z)
    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):

        
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
        return sentence_representation
    
    
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,alpha = None ,**kwargs):
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        semtemce_rep = self.llm_forward(**tokenized_summary)

        logits = self.classifier_forward(semtemce_rep)[0]
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 

        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()
        
    
   
   



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
    
    def forward(self, data_tensor, **kwargs):
        h = F.normalize(data_tensor.float())
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
            # assert q_dims[0] == p_dims[-1], "In and Out dimensions must equal to each other"
            # assert q_dims[-1] == p_dims[0], "Latent dimension for p- and q- network mismatches."
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
    
    def forward(self, data_tensor,**kwargs):
        mu, logvar = self.encode(data_tensor)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    def classifier_forward(self,input, _ ):
        
        mu, logvar = self.encode(input)
        z = self.reparameterize(mu, logvar)
        return self.decode(z) ,mu,logvar
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



def mlp_layers(layer_dims):
    mlp_modules = []
    for i, (d_in, d_out) in enumerate(zip(layer_dims[: -1], layer_dims[1:])):
     
       
        layer = nn.Linear(d_in, d_out)
        nn.init.xavier_normal_(layer.weight)
        nn.init.normal_(layer.bias, std=0.01)
        mlp_modules.append(layer)
        del layer
        if i != len(layer_dims[:-1]) - 1:
            mlp_modules.append(nn.Tanh())

    return nn.Sequential(*mlp_modules)


class MacridVAE(nn.Module):
    def __init__(self,
                 num_items: int,
                 args,
                 layers: list = [400],
                 dropout: float = 0.5,
                 tau_dec: float = 0.1,
                 std: float = 1.,
                 num_iters = 4,
                 reg_weights: List[float] = None,
                 total_anneal_steps: int = 20000):
        super(MacridVAE, self).__init__()
        if reg_weights is None:
            reg_weights = [0.0, 0.0]
        self.num_items = num_items
        self.layers = layers
        self.emb_size = args.dfac
        self.dropout = args.dropout
        self.num_prototypes = args.kfac
        self.num_iters = num_iters
        self.tau = args.tau
        self.tau_dec = tau_dec
        self.nogb = args.nogb
        self.std = std
        self.reg_weights = [args.l2_lambda, args.l2_lambda]
        self.update = 0
        self.total_anneal_steps = total_anneal_steps

        self.encoder_layer_dims = [self.num_items] + self.layers + [self.emb_size * 2]
       
        self.encoder = mlp_layers(self.encoder_layer_dims)

        self.item_embeddings = nn.Embedding(self.num_items, self.emb_size)
        nn.init.xavier_normal_(self.item_embeddings.weight)

        self.prototype_embeddings = nn.Embedding(self.num_prototypes, self.emb_size)
        nn.init.xavier_normal_(self.prototype_embeddings.weight)

        self.dropout_layer = nn.Dropout(self.dropout)
        self.decoder_emb = None
        self.decoder_prot  = None
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            epsilon = torch.zeros_like(std).normal_(mean=0, std=self.std)
            return mu + epsilon * std
        return mu
    def set_item_weights_copy(self):
        self.decoder_emb = deepcopy(self.item_embeddings.weight)
        self.item_embeddings.requires_grad = False
        # self.decoder_emb.requires_grad = False
    def encode(self, input_rating):
        items = F.normalize(self.item_embeddings.weight, dim=1)  # [N, d]
        prototypes = F.normalize(self.prototype_embeddings.weight, dim=1)  # [num_prototypes, emb_size]
        prototypes = prototypes.unsqueeze(dim=0)  # [1, K, d]
        x_latent = prototypes

        mu_list = []
        logvar_list = []
        z_list = []

        cates = None

        for n_iter in range(self.num_iters):
            if self.num_iters > 1 and n_iter == self.num_iters - 1:
                x_latent = x_latent.detach()
            else:
                pass

            cates_logits = x_latent.matmul(items.transpose(0, 1)) / self.tau  # [B/1, K, num_items]
            cates_logits = cates_logits.transpose(1, 2)  # [B/1, num_items, K]

            softmax_dim = -1
            if self.nogb:
                cates = cates_logits.softmax(dim=softmax_dim)
            else:
                cates_sample = F.gumbel_softmax(cates_logits, tau=1, hard=False, dim=softmax_dim)
                cates_mode = cates_logits.softmax(dim=softmax_dim)
                cates = self.training * cates_sample + (1 - self.training) * cates_mode  # [num_items, num_prototypes]

            x_list = []
            for k in range(self.num_prototypes):
                cates_k = cates[:, :, k]  # [B/1, n_items]
                # print(f"{cates_k=}")
                # encoder
                x_k = input_rating * cates_k  # [batch_size, num_items]
                x_k = F.normalize(x_k, p=2, dim=1)  # [batch_size, num_items]
                x_k = self.dropout_layer(x_k)  # [batch_size, num_items]

                if n_iter < self.num_iters - 1:
                    x_latent_k = F.normalize(x_k.matmul(items), dim=-1)
                    x_list.append(x_latent_k.unsqueeze(dim=1))
                else:
                    h = self.encoder(x_k)
                    mu = h[:, : self.emb_size]
                    mu = F.normalize(mu, dim=1)
                    logvar = -h[:, self.emb_size:]

                    mu_list.append(mu)
                    logvar_list.append(logvar)

                    z = self.reparameterize(mu, logvar)
                    # print(f"{z.shape=}")
                    z = F.normalize(z, dim=1)
                    z_list.append(z)

            if n_iter < self.num_iters - 1:
                x_latent = torch.cat(x_list, dim=1)
        

        return {
            'z_list': z_list,
            'mu_list': mu_list,
            'logvar_list': logvar_list,
            'cates_k_list': cates,
            'items': items,
            'prototypes': prototypes
        }

    def decode(self, enc_outputs, need_prob_k=False,z_list = None):
        z_list = torch.stack(enc_outputs['z_list'],dim=1) if z_list is None else z_list
        
        cates_k_list = enc_outputs['cates_k_list'] if self.decoder_prot is None else self.decoder_prot
        items = enc_outputs['items'] if self.decoder_emb is None else F.normalize(self.decoder_emb,dim = -1)

        probs = None
        prob_k_list = []

        for k in range(self.num_prototypes):
            # decoder
            z_k = z_list[:,k,:]
            cates_k = cates_k_list[:, :, k]
            logits_k = z_k.matmul(items.transpose(0, 1)) / self.tau_dec
            probs_k = torch.exp(logits_k)
           
            probs_k = probs_k * cates_k
            if need_prob_k:
                prob_k_list.append(probs_k.unsqueeze(dim=1))
            probs = (probs_k if probs is None else (probs + probs_k))

        return {
            'logits': torch.log(probs),
            'prob_k_list': (torch.cat(prob_k_list, dim=1) if need_prob_k else None),
            'z_list': z_list,
            'mu_list': enc_outputs['mu_list'],
            'logvar_list': enc_outputs['logvar_list']
        }

    def forward(self, input_rating,labels, need_prob_k=False, need_z=False,anneal = 0):
        enc_outputs = self.encode(input_rating)
        dec_outputs = self.decode(enc_outputs, need_prob_k=need_prob_k)
        logits = dec_outputs['logits']
        prob_k_list = dec_outputs['prob_k_list']
        loss = self.calculate_loss(labels, fw_outputs=dec_outputs,anneal = anneal)


        outputs = (logits,loss)
        if need_prob_k:
            outputs['prob_k_list'] = prob_k_list
        if need_z:
            outputs['z_list'] = enc_outputs['z_list']
        return outputs

    def calculate_kl_loss(self, logvar):
        kl_loss = None
        for i in range(self.num_prototypes):
            kl_ = -0.5 * torch.mean(torch.sum(1 + logvar[i] - logvar[i].exp(), dim=1))
            kl_loss = kl_ if kl_loss is None else kl_loss + kl_
        return kl_loss

    @staticmethod
    def calculate_ce_loss(input_rating, logits):
        ce_loss = -(F.log_softmax(logits, 1) * input_rating).sum(dim=1).mean()
        return ce_loss

    def calculate_loss(self, input_rating, fw_outputs=None, anneal = 0 , need_z=False):
      

        if fw_outputs is None:
            fw_outputs = self.forward(input_rating, need_z=need_z)
        kl_loss = self.calculate_kl_loss(fw_outputs['logvar_list'])
        ce_loss = self.calculate_ce_loss(input_rating, fw_outputs['logits'])


        if self.reg_weights[0] != 0 or self.reg_weights[1] != 0:
            if need_z:
                return {
                    'loss': ce_loss + kl_loss * anneal + self.reg_loss(),
                    'z_list': fw_outputs['z_list']
                }
            return  ce_loss + kl_loss * anneal + self.reg_loss()
        if need_z:
            return {
                'loss': ce_loss + kl_loss * anneal,
                'z_list': fw_outputs['z_list']
            }
        return  ce_loss + kl_loss * anneal

    def predict(self, input_rating):
        fw_outputs = self.forward(input_rating=input_rating,
                                  need_prob_k=False)
        prediction = fw_outputs['logits']
        prediction[input_rating.nonzero(as_tuple=True)] = -np.Inf
        return prediction

    def reg_loss(self):
        reg_1, reg_2 = self.reg_weights[: 2]
        loss_1 = reg_1 * self.item_embeddings.weight.norm(2)
        loss_2 = reg_2 * self.prototype_embeddings.weight.norm(2)
        loss_3 = 0

        for name, param in self.encoder.named_parameters():
            if name.endswith('weight'):
                loss_3 = loss_3 + reg_2 * param.norm(2)

        return loss_1 + loss_2 + loss_3

    def predict_per_prototype_outputs(self, input_rating):
        fw_outputs = self.forward(input_rating, need_prob_k=True)
        return fw_outputs['prob_k_list']

    def get_param_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]

    def get_params(self):
        return self.parameters()


class MacridTEARS(T5PreTrainedModel):
    def __init__(self,
                config,epsilon,
                args
                ):
        super().__init__(config)

        self.transformer =   T5EncoderModel(config)
        p_dims = [200,config.num_labels]
        self.epsilon = epsilon
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 800, bias=False),
        )
        self.vae = None
        self.model_parallel = True
        self.args = args
        self.__init__weights()

    def set_vae(self,vae):
        self.vae = vae
    def __init__weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size()[1]
                fan_out = layer.weight.size()[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)

    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
        alpha = .5,
        neg = False
    ) :


        # print(f"{self.vae.modules_to_save['default']=}")
        kfac = self.args.kfac
        split = 400//kfac
        try:
            vae = self.vae.modules_to_save['default']
        except:
            vae = self.vae
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:400],sentence_rep[:,400:]
        #normalize across factors then reshape it back 

        # mu = F.normalize(mu.view(-1,kfac,split),dim=-1).view(-1,400)


        
        z_text = self.reparameterize(mu, logvar)
        z_text = F.normalize(z_text.view(-1,kfac,split),dim=-1).view(-1,400)
        outputs = vae.encode(data_tensor)
        z_rec = outputs['z_list']
        z_rec = torch.cat(z_rec,dim =1 )


        prior_mu = torch.cat(outputs['mu_list'],dim=1)

        prior_logvar = torch.cat(outputs['logvar_list'],dim=1)
    
        z_text = z_text.view(-1,kfac,split)
        z_text = z_text.view(-1,400)
       
        if neg: 

            z_merged =  + (z_rec -z_text)/2
            
        else:

            z_merged = (1-alpha)*z_text + alpha*z_rec 
          
        z_text = z_text.view(-1,kfac,split)
        z_rec = z_rec.view(-1,kfac,split)
        z_merged = F.normalize(z_merged.view(-1,kfac,split),dim=-1)


        
        
        logits_merged = vae.decode(outputs, z_list = z_merged)['logits']
        logits_rec = vae.decode(outputs,z_list = z_rec)['logits']
        logits_text = vae.decode(outputs,z_list = z_text)['logits']
        

        
        

        return logits_merged,logits_rec,logits_text,mu,logvar,prior_mu,prior_logvar,z_rec,None
    
    
    def classifier_forward(self,data_tensor,hidden_states,alpha = None ):
        # hidden_states = self.linear(hidden_states)
        # hidden_states = F.gelu(hidden_states)
        logits = self.classifier(data_tensor,hidden_states)
        return logits


    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
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
        return sentence_representation
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,alpha = .5,neg = False):
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        logits = self.forward(data_tensor = data_tensor,input_ids = tokenized_summary['input_ids'],attention_mask = tokenized_summary['attention_mask'],alpha = alpha,neg = neg)[0]
        #check shape of data_tensor 
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()


class classifierHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size,bias = False)
        self.dropout = nn.Dropout(p=config.dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels,bias = False)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)
   

                
                
    def forward(self, hidden_states):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states



class Classifier(nn.Module):
    def __init__(self,num_labels,hidden_size,dropout):
        super(Classifier,self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size,bias = False)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(hidden_size, num_labels,bias = False)
        self.init_weights()
    def init_weights(self):
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
    def forward(self, hidden_states):

        # hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
        
    

class MistralClassifier(nn.Module):
    def __init__(self,mistral,num_labels,dropout,config,pooling_mode,tokenizer
            
):
        super(MistralClassifier,self).__init__()
        self.config = config
        self.mistral = mistral
        self.dropout = nn.Dropout(dropout)
        self.pooling_mode = pooling_mode
        self.tokenizer = tokenizer
        self.classifier = Classifier(num_labels,4096,dropout).bfloat16()
    def forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                labels_tr: Optional[torch.LongTensor] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
  

        hidden_states = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, labels=labels, use_cache=use_cache, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict)
        
        logits = self.classifier_forward(hidden_states)


        return [logits]
    def classifier_forward(self,hidden_states):
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits
    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,):
        hidden_states = self.mistral(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        features = {"input_ids": input_ids, "attention_mask": attention_mask}
        hidden_states = self.get_pooling(features,hidden_states)
        return hidden_states
    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ):
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        semtemce_rep = self.llm_forward(**tokenized_summary)

        logits = self.classifier_forward(semtemce_rep)
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()


    def get_pooling(self, features, last_hidden_states):  # All models padded from left
        assert self.tokenizer.padding_side == 'left', "Pooling modes are implemented for padding from left."

        seq_lengths = features["attention_mask"].sum(dim=-1)
        if self.pooling_mode == "mean":
            return torch.stack([last_hidden_states[i, -length:, :].mean(dim=0) for i, length in enumerate(seq_lengths)], dim=0) 
        elif self.pooling_mode == "weighted_mean":
            bs, l, _ = last_hidden_states.shape
            complete_weights = torch.zeros(bs, l, device=last_hidden_states.device)
            for i, seq_l in enumerate(seq_lengths):
                if seq_l > 0:
                    complete_weights[i, -seq_l:] = torch.arange(seq_l) + 1
                    complete_weights[i] /= torch.clamp(complete_weights[i].sum(), min=1e-9)
            return torch.sum(last_hidden_states * complete_weights.unsqueeze(-1), dim=1)
        elif self.pooling_mode == "eos_token" or self.pooling_mode == "last_token":
            return last_hidden_states[:, -1]
        elif self.pooling_mode == "bos_token":
            return last_hidden_states[features["input_ids"]==self.tokenizer.bos_token_id]
        else:
            raise ValueError(f"{self.pooling_mode} is not implemented yet.")
        

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


class EASY(nn.Module):
    def __init__(self, num_items, l2_reg=5000):
        super(EASY, self).__init__()
        self.num_items = num_items
        self.l2_reg = l2_reg
        self.B = None

        
    def diag_indices_f(self,n):
        rows = torch.arange(n)
        cols = torch.arange(n)
        return rows, cols
    def forward(self, input):
        # If B matrix is not computed, compute it

        if self.B is None:
            G = input.T @ input
            diag_indices = self.diag_indices_f(G.shape[0])
            G[diag_indices[0], diag_indices[1]] += self.l2_reg
            P = torch.inverse(G)
            self.B = P / (-torch.diag(P))
            self.B[diag_indices[0], diag_indices[1]] = 0

        # Compute transformed input
        transformed_input = input @ self.B

        return transformed_input
    def train(self, input):
        self.B = None
        return self.forward(input)
    def eval(self,in_items,labels): 
        from collections import defaultdict
        metrics = defaultdict(list)
        
        recon = in_items @ self.B
        recon[np.where(in_items > 0)] = -1e-20
        recon = recon.cpu().numpy()
        labels = labels.cpu().numpy()
        k_values = [10,20,50]
        for k in k_values:
            metrics[f'ndcg@{k}'].append(NDCG_binary_at_k_batch(recon, labels, k=k).tolist())
            # metrics[f'mrr@{k}'].append(MRR_at_k(recon, labels, k=k,mean = False).tolist())
            metrics[f'recall@{k}'].append(Recall_at_k_batch(recon, labels, k=k,mean = False).tolist())
        for key in metrics.keys():
            metrics[key] = np.mean(sum(metrics[key],[]))


        return metrics



def get_tokenizer(args):

    if args.embedding_module == 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp':
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_module)
    else:
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-base')
    return tokenizer

def get_model(args, tokenizer, num_movies, rank, world_size):
    if args.embedding_module == 'google-t5/t5-3b' or args.embedding_module == 'google-t5/t5-large':
        model,lora_config = get_google_t5_model(args, num_movies,args.embedding_module)
    elif args.embedding_module == "microsoft/phi-2":
        model,lora_config = get_microsoft_phi_model(args, num_movies, tokenizer)
    elif args.embedding_module == "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp":
        model,lora_config = get_mcgill_model(args, num_movies, tokenizer)
    elif args.embedding_module == "t5_film":
        model,lora_config  = get_VAE_model(args,num_movies)
    elif args.embedding_module =='VAE':
        if args.DAE:
            return get_MultiDAE(args,num_movies)
            

        model,lora_config = get_MultiVAE(args,num_movies)
    elif args.embedding_module =='sentenceT5ClassificationPooling':
        model,lora_config = get_t5_pooling(args,num_movies )
    elif args.embedding_module =='T5Vae':
        
        model,lora_config = get_t5VAE(args,num_movies )
    elif args.embedding_module == 'VariationalT5':
        model, lora_config = get_VariationalT5(args,num_movies)
    elif args.embedding_module == 'OTVae':
        model,lora_config = get_OT(args,num_movies)
    elif args.embedding_module == 'RecVAE':
        return get_RecVAE(args,num_movies)
    elif args.embedding_module == 'OTRecVAE':
        return get_t5RecVAE(args,num_movies)
    elif args.embedding_module =='FT5RecVAE':
        return get_OT_RecVAE(args,num_movies)
    elif args.embedding_module == 'MultiDAE':
        return get_MultiDAE(args,num_movies)
    elif args.embedding_module == 'MacridVAE':
        return get_MacridVAE(args,num_movies)
    elif args.embedding_module =='MacridTEARS':
        return get_MacridTEARS(args,num_movies)
    else:
        raise ValueError(f"Unsupported embedding module: {args.embedding_module}")
    return model,lora_config

def get_VariationalT5(args,num_movies):
    model = sentenceT5Vae.from_pretrained('google-t5/t5-base', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classifier'])
    return model,lora_config

def get_VAE_model(args,num_movies):
    model = sentenceT5ClassificationVAE.from_pretrained('google-t5/t5-3b', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classifier'])
    
    return model,lora_config
def get_t5_pooling(args,num_movies):
    model = sentenceT5ClassificationPooling.from_pretrained('google-t5/t5-3b', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classification_head','VAE','mlp'])

    return model,lora_config
    
def get_MultiVAE(args,num_movies):
    pdims = [400,num_movies]
    model =  MultiVAE(pdims, dropout = args.dropout)
    return model,None
def get_MacridVAE(args,num_movies):
    args.dfac = 400//args.kfac
    print(f"{args.dfac=}")
    model = MacridVAE(num_movies,args)

    return model,None

def get_MultiDAE(args,num_movies):
    pdims = [400,num_movies]
    model =  MultiDAE(pdims, dropout = args.dropout)
    return model,None

def get_google_t5_model(args, num_movies,model):
            
    model = sentenceT5Classification.from_pretrained('google-t5/t5-small', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classification_head'])
    return model,lora_config
def get_EASE(args,num_movies,l2_reg):
    model = EASY(num_movies,l2_reg)
    return model





def get_mcgill_model(args, num_movies, tokenizer):

        config = AutoConfig.from_pretrained(args.embedding_module, trust_remote_code=True)
        bnb_config = BitsAndBytesConfig(  
                load_in_4bit= True,
                bnb_4bit_quant_type= "nf4",
                bnb_4bit_compute_dtype= torch.bfloat16,
                bnb_4bit_use_double_quant= True,
            )
        model = AutoModel.from_pretrained(
            args.embedding_module,
            trust_remote_code=True,
            config=config,
            quantization_config=bnb_config,
            # torch_dtype=torch.bfloat16,
            # device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        model = PeftModel.from_pretrained(
            model,
            "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp"
        )
        model = model.merge_and_unload()  # This can take several minutes on cpu
        
        model = PeftModel.from_pretrained(
            model, "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp-unsup-simcse"
        )

        model_config = model.config
        model = MistralClassifier(model, num_movies, args.dropout,config,pooling_mode = args.pooling,tokenizer = tokenizer)
        # model = model.bfloat16()
        # model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing = True)
        #check type of all the parameters in the model if any equal torckuin8 break the loop and print it 
        # for name, param in model.named_parameters():
        #     if param.dtype == torch.uint8:
        #         print(f"{name} has dtype torch.uint8")
        #         print(f"{param=}")
        #         exit()
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                                target_modules=["q_proj", "k_proj", "v_proj"],
                                bias = 'none',
                                
                                modules_to_save=['classifier'])
        return model,lora_config

def get_t5VAE(args,num_movies):
    pdims = [400,num_movies]
    prior =  MultiVAE(pdims, dropout = args.dropout)
    if args.data_name =='ml-1m':
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_ml-1m_embedding_module_VAE_2024-04-27_21-08-15.csv.pt'
    elif args.data_name =='netflix':
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_netflix_embedding_module_VAE_2024-05-02_14-27-25.csv.pt'
    state_dict = torch.load(p)
    prior.load_state_dict(state_dict)
    model = T5Vae.from_pretrained('google-t5/t5-base',num_labels=num_movies, classifier_dropout = args.dropout,prior = prior,epsilon = args.epsilon)

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            # modules_to_save=['classification_head','prior','mlp'])
                            modules_to_save=['classification_head','mlp','vae'])
    # model.prior.train()
    #turn off the gradient for the prior.q layers

    return model ,lora_config

def get_RecVAE(args,num_movies):
    pdims = [400,num_movies]

    
    model =  RecVAE(pdims, dropout = args.dropout,gamma = args.gamma)
    # for name,param in model.named_parameters():
    #     print(f"{name,param.mean()=}")
    # exit()
        
    return model,None

def get_t5RecVAE(args,num_movies): 
    pdims = [400,num_movies]
    # print(f"{num_movies=}")

    prior =  RecVAE(pdims, dropout = args.dropout,gamma= args.gamma)
    if args.data_name == 'netflix':
        # print('netflix')
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_netflix_embedding_module_RecVAE_2024-05-02_14-54-48.csv.pt'
    elif args.data_name =='ml-1m':
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_ml-1m_embedding_module_RecVAE_2024-05-02_13-10-49.csv.pt'
        # p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_ml-1m_embedding_module_RecVAE_2024-04-27_15-53-38.csv.pt'
        
        
    state_dict = torch.load(p)
    prior.load_state_dict(state_dict)
    # classifier =  RecVAE(pdims, dropout = args.dropout)

    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            # modules_to_save=['classification_head','prior','mlp'])
                            modules_to_save=['classification_head','mlp','RecVAE','Encoder','CompositePrior','concat_mlp'])
    model = T5Vae.from_pretrained('google-t5/t5-base',num_labels=num_movies, 
                                  classifier_dropout = args.dropout,prior = None,
                                  epsilon = args.epsilon,concat = args.concat)
    model = get_peft_model(model, lora_config)
    model.set_vae( prior )
    # print(f"{model.vae=}")
    # model.prior.train()
    

    return model ,None
    
def get_MacridTEARS(args,num_movies): 
    args.dfac = 400//args.kfac
    
    vae = MacridVAE(num_movies,args)
    if args.data_name =='ml-1m':
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_ml-1m_embedding_module_MacridVAE_2024-05-10_15-07-31.csv.pt'
    elif args.data_name == 'netflix':
        
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_netflix_embedding_module_MacridVAE_2024-05-11_11-36-07.csv.pt'
        

    vae.load_state_dict(torch.load(p))
    vae.set_item_weights_copy()
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            # modules_to_save=['classification_head','prior','mlp'])
                       
                            modules_to_save=['classification_head','mlp','RecVAE','Encoder','CompositePrior','concat_mlp'])
    model = MacridTEARS.from_pretrained('google-t5/t5-base',args = args,num_labels=num_movies, 
                                  
                                  epsilon = args.epsilon)
    
    model = get_peft_model(model, lora_config)
    model.set_vae( vae )
    return model ,None
    

def get_OT_RecVAE(args,num_movies):
 
    pdims = [400,num_movies]


    prior =  RecVAE(pdims, dropout = args.dropout,gamma= args.gamma)
    if args.data_name == 'netflix':

        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_netflix_embedding_module_RecVAE_2024-04-27_10-56-43.csv.pt'
    else: 
        
        p = f'{args.scratch}/saved_model/{args.data_name}/t5_classification_fixed_data_ml-1m_embedding_module_RecVAE_2024-04-27_20-51-00.csv.pt'
        
        
    state_dict = torch.load(p)
    prior.load_state_dict(state_dict)
    
    llm = sentenceT5Vae.from_pretrained('google-t5/t5-base', num_labels=num_movies, classifier_dropout = args.dropout)
    
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classifier'])
    llm = get_peft_model(llm, lora_config)
    if args.data_name =='ml-1m':
        
        state_dict_llm = torch.load(f'{args.scratch}/saved_model/{args.data_name}/ot_train_vae_ml-1m_embedding_module_VariationalT5_2024-04-28_12-23-19.csv.pt')
    else:
        state_dict_llm = torch.load(f'{args.scratch}/saved_model/{args.data_name}/ot_train_vae_ml-1m_embedding_module_VariationalT5_2024-04-28_12-23-19.csv.pt')

    # for key in state_dict_llm.keys():
    #     if 'classifier' in  key:
    #         new_key = key.replace('classifier', 'vae')
    #         state_dict_llm[new_key] = state_dict_llm.pop(key)
    llm.load_state_dict(state_dict_llm)

    llm.merge_and_unload()
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            # modules_to_save=['classification_head','prior','mlp'])
                            modules_to_save=['classifier'])
    llm = get_peft_model(llm, lora_config)
    #turn off the gradient for the prior.q layers
    model = otVAE.from_pretrained('google-t5/t5-base',num_labels=num_movies, classifier_dropout = args.dropout,vae = prior,epsilon = args.epsilon, llm = llm,prior = prior.prior)




    return model ,None

