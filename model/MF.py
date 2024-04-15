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
from transformers import AutoModel
from peft import LoraConfig, TaskType, PeftModel,get_peft_model
from transformers import T5Tokenizer ,AutoTokenizer


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
        output_attentions = None,
        output_hidden_states = None,
        return_dict_in_generate = None,
    ) :
        sentence_rep = self.llm_forward(input_ids=input_ids, attention_mask=attention_mask, inputs_embeds=inputs_embeds, return_dict=return_dict)
        logits = self.classifier_forward(sentence_rep)
        
        
        return Seq2SeqSequenceClassifierOutput(

            logits=logits
        )
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

class PhiPreTrainedModel(PreTrainedModel):
    config_class = PhiConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


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

class PhiForSequenceClassification(PhiPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = PhiModel(config)
        self.act = torch.functional.F.tanh
        self.score = classifierHead(config)

        # Initialize weights and apply final processing
        # self.post_init()

    def get_input_embeddings(self):
        return self.model.embd.wte#changed

    def set_input_embeddings(self, value):
        self.model.embd.wte = value#changed


    def forward(
        self,
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
    ) -> Union[Tuple, SequenceClassifierOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        model_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,

        )
        hidden_states = model_outputs[0]#changed
        logits = self.score(hidden_states)
        # print(logits)

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        if self.config.pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if self.config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids, self.config.pad_token_id).int().argmax(-1) - 1).to(
                    logits.device
                )
            else:
                sequence_lengths = -1

        pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        loss = None
        if labels is not None:
            labels = labels.to(logits.device)
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

        if not return_dict:
            output = (pooled_logits,) + model_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutputWithPast(
            logits=pooled_logits,
            loss=loss,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )#changed


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



def get_tokenizer(args):
    if args.embedding_module == 'google-t5/t5-3b':
        tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-3b')
    elif args.embedding_module == 'McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp':
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_module)
    else: 
        tokenizer = AutoTokenizer.from_pretrained(args.embedding_module,padding_side= 'left' if args.embedding_module == 'google-t5/t5-3b' else "right",
                                               add_eos_token=False if args.embedding_module == 'google-t5/t5-3b' else True,  
                                               add_bos_token=False if args.embedding_module == 'google-t5/t5-3b' else True,
                                               use_fast=True if args.embedding_module == 'google-t5/t5-3b' else False)
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_model(args, tokenizer, num_movies, rank, world_size):
    if args.embedding_module == 'google-t5/t5-3b':
        model,lora_config = get_google_t5_model(args, num_movies)
    elif args.embedding_module == "microsoft/phi-2":
        model,lora_config = get_microsoft_phi_model(args, num_movies, tokenizer)
    elif args.embedding_module == "McGill-NLP/LLM2Vec-Mistral-7B-Instruct-v2-mntp":
        model,lora_config = get_mcgill_model(args, num_movies, tokenizer)
    else:
        raise ValueError(f"Unsupported embedding module: {args.embedding_module}")
    return model,lora_config

def get_google_t5_model(args, num_movies):
    model = sentenceT5Classification.from_pretrained('google-t5/t5-3b', num_labels=num_movies, classifier_dropout = args.dropout)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=['q','v','k'],
                            modules_to_save=['classification_head'])

    return model,lora_config

def get_microsoft_phi_model(args, num_movies, tokenizer):
    configuration = AutoConfig.from_pretrained(args.embedding_module)
    configuration.dropout = args.dropout
    configuration.num_labels = num_movies  # Set the number of labels here
    model = PhiForSequenceClassification.from_pretrained(args.embedding_module, torch_dtype=torch.float32, config=configuration)
    lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                            target_modules=["q_proj", "k_proj", "v_proj"],
                            modules_to_save=['score'])
    model = get_peft_model(model, lora_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    if args.warmup> 0 :
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = False
    return model

def get_mcgill_model(args, num_movies, tokenizer):

        config = AutoConfig.from_pretrained(args.embedding_module, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            args.embedding_module,
            trust_remote_code=True,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
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
        lora_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.dropout,
                                target_modules=["q_proj", "k_proj", "v_proj"],
                                
                                modules_to_save=['classifier'])
        return model,lora_config

