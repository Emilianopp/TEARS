from helper.eval_metrics import Recall_at_k_batch,NDCG_binary_at_k_batch,MRR_at_k
import sys 
sys.path.append('../')
import os
from collections import defaultdict
import pandas as pd 
from helper.dataloader import map_id_to_genre
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers.models.t5.modeling_t5 import T5PreTrainedModel,T5EncoderModel
from peft import LoraConfig, TaskType,get_peft_model,prepare_model_for_kbit_training
from transformers import T5Tokenizer 
import ot 

from typing import List, Optional
from copy import deepcopy



class MultiDAE(nn.Module):

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

    def __init__(self, p_dims, q_dims=None, dropout=0.5):
        super(MultiVAE, self).__init__()
        self.p_dims = p_dims
        if q_dims:

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
    
    def forward(self, data_tensor,lables=None,**kwargs):
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

    def forward(self, input_rating,labels, need_prob_k=False, need_z=False,anneal = 0, **kwargs):
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


class EASE(nn.Module):
    def __init__(self, num_items, l2_reg=5000):
        super(EASE, self).__init__()
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
    @staticmethod
    def train(args,train_dataloader,val_dataloader,test_dataloader,num_movies):
        train_items = []
        target_items = []
        for b in train_dataloader: 
            train_items.append(b['labels_tr'])
            target_items.append(b['labels'])
        train_items = torch.concat(train_items)
        target_items = torch.concat(target_items)
        
        val_items = []
        val_target_items = []
        for b in val_dataloader: 
            val_items.append(b['labels_tr'])
            val_target_items.append(b['labels'])
        val_target_items = torch.concat(val_target_items)
        val_items = torch.concat(val_items)
        
        test_items = []
        test_target_items = []
        for b in test_dataloader: 
            test_items.append(b['labels_tr'])
            test_target_items.append(b['labels'])
        test_target_items = torch.concat(test_target_items)
        test_items = torch.concat(test_items)
        train_matrix = 0
        l2_regs = np.linspace(1,10000,50)
        for l2_reg in (pbar:=tqdm(l2_regs[1:2])):
            model = get_EASE(args,num_movies,l2_reg)
            model(target_items)
            metrics = model.eval(val_items,val_target_items)
            recall = metrics['ndcg@50']
            pbar.set_description(f"Recall@50 = {recall}, l2_reg = {l2_reg}")
            if recall > train_matrix:
                train_matrix = recall
                best_l2_reg = l2_reg
        
        model = get_EASE(args,num_movies,best_l2_reg)
        metrics = model.eval(test_items,test_target_items)
        #logging
        csv_path = f"./model_logs/{args.data_name}/{args.embedding_module}/parameter_sweep.csv"
        log_results = pd.DataFrame({'model_name': 'EASE',**metrics}).round(4)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        log_results.to_csv(csv_path)
        print(f'Saved to {csv_path}')




class Tears(T5PreTrainedModel):
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)

class TearsBase(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        self.classifier = MultiVAE(q_dims = [config.d_model,800,400],p_dims = [400,config.num_labels],dropout = config.classifier_dropout)
        self.model_parallel = True
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs
    ) :
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        logits,mu,logvar = self.classifier(sentence_rep)
        return logits,mu,logvar

    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None,
                **kwargs):

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
        logits = self.classifier(semtemce_rep)[0]
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 

        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()




# TEARS VAE
class TearsVAE(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,config,epsilon,args):
        super().__init__(config)
        self.transformer =   T5EncoderModel(config)
        self.epsilon = epsilon
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 800, bias=False),
        )
        self.model_parallel = True
        self.args = args
        self.__init__weights()

    def __init__weights(self):
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                fan_in = layer.weight.size()[1]
                fan_out = layer.weight.size()[0]
                std = np.sqrt(2.0 / (fan_in + fan_out))
                layer.weight.data.normal_(0.0, std)
    
    def set_vae(self,vae):
        self.vae = vae
    def classifier_forward(self,data_tensor,hidden_states,alpha = None ):
        logits = self.classifier(data_tensor,hidden_states)
        return logits

    def llm_forward(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_dict: Optional[bool] = None,**kwargs):
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
    

    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        alpha = .5,
        neg = False,
        **kdwargs
    ) :

        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:400],sentence_rep[:,400:]
        z_text = self.vae.reparameterize(mu, logvar)
        rec_mu, rec_logvar = self.vae.encode(data_tensor)
        z_rec = self.vae.reparameterize(rec_mu, rec_logvar)


        if neg: 

            z_merged =  + (z_rec -z_text)/2
            
        else :

            z_merged = (1-alpha)*z_text + alpha*z_rec 

        logits_merged = self.vae.decode(z_merged)
        logits_rec = self.vae.decode(z_rec)
        logits_text = self.vae.decode(z_text)
        return logits_merged,logits_rec,logits_text,mu,logvar,rec_mu,rec_mu

    def generate_recommendations(self,summary,tokenizer,data_tensor,topk,rank = 0 ,alpha = .5,neg = False,return_emb = False):
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        if return_emb:
            tokenized_summary = tokenizer(summary,return_tensors="pt")
            tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
            
            return self.forward(data_tensor = data_tensor,input_ids = tokenized_summary['input_ids'],attention_mask = tokenized_summary['attention_mask'],alpha = alpha,neg = neg)
        tokenized_summary = tokenizer([summary],return_tensors="pt")

        
        tokenized_summary = {k: v.to(rank) for k, v in tokenized_summary.items()}
        logits = self.forward(data_tensor = data_tensor,input_ids = tokenized_summary['input_ids'],attention_mask = tokenized_summary['attention_mask'],alpha = alpha,neg = neg)[0]
        #check shape of data_tensor 
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk
        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist() 




class MacridTEARS(TearsVAE):
    def __init__(self,
                config,epsilon,
                args
                ):
        super().__init__(config,epsilon,args)
    
    #overwrite forwrard function to align with MacrivVAE specific configurations
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        alpha = .5,
        neg = False,
        **kwargs
    ) :

        #have to split the dimensions to be compatible with MacridVAE factors
        kfac = self.args.kfac
        split = self.args.emb_dim//kfac
        
        #get sentence representation
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:self.args.emb_dim],sentence_rep[:,self.args.emb_dim:]

        
        #[B,emb_dim]
        z_text = self.vae.reparameterize(mu, logvar)
        outputs = self.vae.encode(data_tensor)
    
        #MacridVAE's output is normalized across factors so we normalize as well
        z_text = F.normalize(z_text.view(-1,kfac,split),dim=-1).view(-1,self.args.emb_dim)
        z_rec = outputs['z_list']
        z_rec = torch.cat(z_rec,dim =1 )
        
        #get back into shape [B,emb_dim] for pooling
        rec_mu = torch.cat(outputs['mu_list'],dim=1)
        rec_logvar = torch.cat(outputs['logvar_list'],dim=1)

        #pooling       
        if neg: 
            z_merged =  + (z_rec -z_text)/2

        else:
            z_merged = (1-alpha)*z_text + alpha*z_rec 
          
        #reshape to [B,kfac,split]
        z_text = z_text.view(-1,kfac,split)
        z_rec = z_rec.view(-1,kfac,split)
        z_merged = F.normalize(z_merged.view(-1,kfac,split),dim=-1)


        
        #use shared decoder
        logits_merged = self.vae.decode(outputs, z_list = z_merged)['logits']
        logits_rec = self.vae.decode(outputs,z_list = z_rec)['logits']
        logits_text = self.vae.decode(outputs,z_list = z_text)['logits']
    
        return logits_merged,logits_rec,logits_text,mu,logvar,rec_mu,rec_logvar
    



class GenreModel(nn.Module):
    def __init__():
        super().__init__()
    def set_vae(self,vae):
        self.vae = vae
    def make_genre_vector(self, input ): 
        # Initialize the genre vector
        genre_vector = torch.zeros(input.shape[0], self.num_genres, device=input.device)
        # Create a genre map tensor for quick look-up
        genre_map_tensor = torch.tensor([[int(g in self.genre_map[j]) for g in sorted(self.genres_l)] for j in range(input.shape[1])], device=input.device).float()
        genre_vector += input @ genre_map_tensor
        #calcualte proportions
        normalized_genre_vector = genre_vector / genre_vector.sum(dim=1).unsqueeze(1)

        return normalized_genre_vector
    def generate_recommendations(self,data_tensor,topk,alpha = .5,neg = False,mask_genre = None):

        data_tensor=data_tensor.unsqueeze(0)

        sentence_rep = self.make_genre_vector(data_tensor)
        if mask_genre is not None:

            mask_index = self.genre_inds[mask_genre]
            mask = torch.zeros(sentence_rep.size(1), dtype=torch.bool)
            mask[mask_index] = True
            sentence_rep[:,~mask] = 0
            sentence_rep[:,mask] = 1

        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:self.args.emb_dim],sentence_rep[:,self.args.emb_dim:]
        z_text = self.reparameterize(mu, logvar)
        prior_mu, prior_logvar = self.vae.encode(data_tensor)
        z_rec = self.vae.reparameterize(prior_mu, prior_logvar)
        if neg: 

            z_merged =  + (z_rec -z_text)/2
            
        else:

            z_merged = (1-alpha)*z_text + alpha*z_rec 


        logits = self.vae.decode(z_merged)

        #check shape of data_tensor 
        
        #mask the logits of the data_tensor 
        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist() 
        



class GenreVae(nn.Module):
    _keys_to_ignore_on_load_unexpected = ["decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight"]
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    def __init__(self,num_genres,prior,genre_map,genres_l,epsilon,concat=False,num_movies = None):
        super().__init__()
        self.epsilon = epsilon
        self.num_genres = num_genres
        self.mlp = nn.Sequential(
            nn.Linear( num_genres,800, bias=False),
        )
        self.vae = prior
        self.genres_l = genres_l
        self.model_parallel = True
        self.concat = concat
        if concat: 
            self.concat_mlp = nn.Linear(800,self.args.emb_dim,bias = False)
        self.__init__weights()
        self.genre_map = genre_map 
        self.genre_coutns = {g:0 for g in genres_l}
        self.genre_inds = {g.replace('-',' '):i for i,g in enumerate(sorted(genres_l))}
        for g in genre_map:
            for g_ in genre_map[g]:
                self.genre_coutns[g_] += 1
        self.genre_count_vector = torch.tensor([self.genre_coutns[g] for g in genres_l]).float()
        self.genre_columns_one_hot = defaultdict(lambda : torch.zeros(num_movies))
        
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
            
    
    def forward(
        self,
        data_tensor: torch.LongTensor = None,
        alpha = .5,
        neg = False,
        **kwargs
    ) :
        sentence_rep = self.make_genre_vector(data_tensor)
        sentence_rep = self.mlp(sentence_rep)
        mu,logvar = sentence_rep[:,:self.args.emb_dim],sentence_rep[:,self.args.emb_dim:]
        z_text = self.vae.reparameterize(mu, logvar)
        prior_mu, prior_logvar = self.vae.encode(data_tensor)
        z_rec = self.vae.reparameterize(prior_mu, prior_logvar)
        if neg: 

            z_merged =  + (z_rec -z_text)/2
            
        else:

            z_merged = (1-alpha)*z_text + alpha*z_rec 


        logits_merged = self.vae.decode(z_merged)
        logits_rec = self.vae.decode(z_rec)
        logits_text = self.vae.decode(z_text)
        return logits_merged,logits_rec,logits_text,mu,logvar,prior_mu,prior_logvar,z_rec,None
    

 
class GenreTEARS(GenreModel):
    
    def __init__(self,num_genres,genre_map,genres_l,epsilon,dropout,args,concat=False,num_movies = None):
        super().__init__()
        self.epsilon = epsilon
        self.num_genres = num_genres
        self.mlp = nn.Sequential(
            nn.Linear( num_genres,800, bias=False),
        )
        self.genres_l = genres_l
        self.model_parallel = True
        self.concat = concat
        if concat: 
            self.concat_mlp = nn.Linear(800,self.args.emb_dim,bias = False)

        self.genre_map = genre_map 
        self.genre_coutns = {g:0 for g in genres_l}
        #fill in genre counts 
        self.genre_inds = {g.replace('-',' '):i for i,g in enumerate(sorted(genres_l))}
        for g in genre_map:
            for g_ in genre_map[g]:
                self.genre_coutns[g_] += 1
        
        self.genre_count_vector = torch.tensor([self.genre_coutns[g] for g in genres_l]).float()
        self.genre_columns_one_hot = defaultdict(lambda : torch.zeros(num_movies))
        self.classifier = MultiVAE(q_dims = [num_genres,800,self.args.emb_dim],p_dims = [self.args.emb_dim,num_movies],dropout = dropout)
        self.model_parallel = True
        

    def forward(
        self,
        data_tensor = None,
        **kwargs
    ) :
        sentence_rep = self.make_genre_vector(data_tensor)
        logits,mu,logvar = self.classifier(sentence_rep)

        
        return logits,mu,logvar

    def generate_recommendations(self,data_tensor,topk,mask_genre = None):

        data_tensor=data_tensor.unsqueeze(0)
      
        sentence_rep = self.make_genre_vector(data_tensor)
        
        if mask_genre is not None:
            mask_index = self.genre_inds[mask_genre]
            mask = torch.zeros(sentence_rep.size(1), dtype=torch.bool)
            mask[mask_index] = True
            sentence_rep[:,~mask] = 0
            sentence_rep[:,mask] = 1

        logits = self.classifier_forward(sentence_rep)[0]
        #check shape of data_tensor 
        if len(data_tensor.shape) == 1:
            data_tensor = data_tensor.unsqueeze(0)
        
        #mask the logits of the data_tensor 

        logits[data_tensor >= 1] = np.log(.000001)
        
        #returned ranked items based on topk

        ranked_logts = torch.topk(logits,topk)


        return ranked_logts.indices.flatten().tolist()
           
        



def get_tokenizer(args):
    return T5Tokenizer.from_pretrained('google-t5/t5-base')

def get_model(args, num_movies):

    if args.embedding_module =='MVAE':


        model = get_MultiVAE(args,num_movies)

    elif args.embedding_module =='TearsMVAE':
        model = get_TearsVAE(args,num_movies )
    elif args.embedding_module == 'TearsBase':
        model = get_TearsBase(args,num_movies)
    elif args.embedding_module == 'RecVAE':
        return get_RecVAE(args,num_movies)
    elif args.embedding_module == 'TearsRecVAE':
        return get_t5RecVAE(args,num_movies)
    elif args.embedding_module == 'MultiDAE':
        return get_MultiDAE(args,num_movies)
    elif args.embedding_module == 'MacridVAE':
        return get_MacridVAE(args,num_movies)
    elif args.embedding_module =='TearsMacrid':
        return get_MacridTEARS(args,num_movies)
    elif args.embedding_module == 'RecVAEGenreVAE':
        return get_GenreVAE(args,num_movies)
    elif args.embedding_module == 'GenreTEARS':
        return get_GenreTEARS(args,num_movies)
    else:
        raise ValueError(f"Unsupported embedding module: {args.embedding_module}")
    return model

    
def get_MultiVAE(args,num_movies):
    pdims = [args.emb_dim,num_movies]
    model =  MultiVAE(pdims, dropout = args.dropout)
    return model

def get_MacridVAE(args,num_movies):
    args.dfac = args.emb_dim//args.kfac
    model = MacridVAE(num_movies,args)

    return model

def get_MultiDAE(args,num_movies):
    pdims = [args.emb_dim,num_movies]
    model =  MultiDAE(pdims, dropout = args.dropout)
    return model

def get_EASE(args,num_movies,l2_reg):
    model = EASE(num_movies,l2_reg)
    return model

def get_TearsBase(args,num_movies):
    model = TearsBase.from_pretrained('google-t5/t5-base', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classifier'])
    get_peft_model(model, lora_config)
    return model


def get_RecVAE(args,num_movies):
    pdims = [args.emb_dim,num_movies]
    model =  RecVAE(pdims, dropout = args.dropout,gamma = args.gamma)
    return model


def get_TearsVAE(args,num_movies):
    pdims = [args.emb_dim,num_movies]
    vae =  MultiVAE(pdims, dropout = args.dropout)
    state_dict = torch.load(p)
    p = f'{args.scratch}/saved_model/{args.data_name}/{args.vae_path}'
    vae.load_state_dict(state_dict)
    model = TearsVAE.from_pretrained('google-t5/t5-base',num_labels=num_movies, epsilon = args.epsilon,args = args)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classification_head','mlp','vae'])
    model = get_peft_model(model, lora_config)
    model.set_vae(vae)

    return model 


def get_t5RecVAE(args,num_movies): 
    pdims = [args.emb_dim,num_movies]
    prior =  RecVAE(pdims, dropout = args.dropout,gamma= args.gamma)
    p = f'{args.scratch}/saved_model/{args.data_name}/{args.vae_path}'
    state_dict = torch.load(p)
    prior.load_state_dict(state_dict)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classification_head','mlp','RecVAE','Encoder','CompositePrior','concat_mlp'])
    model = TearsVAE.from_pretrained('google-t5/t5-base',num_labels=num_movies, 
                                  args = args,
                                  epsilon = args.epsilon)
    model = get_peft_model(model, lora_config)
    model.set_vae( prior )

    return model 


def get_MacridTEARS(args,num_movies): 
    args.dfac = args.emb_dim//args.kfac
    vae = MacridVAE(num_movies,args)
    p = f'{args.scratch}/saved_model/{args.data_name}/{args.vae_path}'
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
    return model 

def get_GenreTEARS(args,num_movies):
    id_to_genre = map_id_to_genre(args.data_name)
    genre_s = set()
    for key in id_to_genre.keys():
        [genre_s.add(g) for g in id_to_genre[key]]
    num_genres = len(genre_s)
    model = GenreTEARS(num_genres=num_genres,
                       genres_l = list(genre_s),
                       num_movies=num_movies,genre_map = id_to_genre,epsilon = args.epsilon,dropout= args.dropout)
    return model

def get_GenreVAE(args,num_movies): 
    pdims = [args.emb_dim,num_movies]
    prior =  RecVAE(pdims, dropout = args.dropout,gamma= args.gamma)
    p = f'{args.scratch}/saved_model/{args.data_name}/{args.vae_path}'
    state_dict = torch.load(p)
    prior.load_state_dict(state_dict)
    id_to_genre = map_id_to_genre(args.data_name)
    genre_s = set()
    for key in id_to_genre.keys():
        [genre_s.add(g) for g in id_to_genre[key]]
    num_genres = len(genre_s)
    model = GenreVae(num_genres=num_genres, 
                                  genre_map=id_to_genre,prior = None,
                                  genres_l = list(genre_s), epsilon = args.epsilon,concat = args.concat,num_movies = num_movies)
    model.set_vae( prior )
    

    return model 


    
