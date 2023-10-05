import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import spatial
from .decoderMLP import decoderMLP
import time

def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
        
class MatrixFactorization(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(MatrixFactorization, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.output_emb

        self.user_embeddings = nn.Embedding(self.num_users, args.output_emb).to(self.device)
        self.item_embeddings = nn.Embedding(self.num_items, args.output_emb).to(self.device)
        self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        self.myparameters = [self.user_embeddings.weight, self.item_embeddings.weight]

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -torch.log( nn.Sigmoid()(tmp) )

        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def forward(self, user, pos, neg):
        # thr: threshold

        user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb, pos_emb, neg_emb = \
            self.user_embeddings(user_id), self.item_embeddings(pos_id), self.item_embeddings(neg_id)

        return user_emb, pos_emb, neg_emb

    def predict(self, user_id):
        user_emb = self.user_embeddings(user_id)
        pred = user_emb.mm(self.item_embeddings.weight.t())

        return pred


    def get_embeddings(self, ids, emb_name):
        if emb_name == 'user':
            return self.user_embeddings[ids]
        elif emb_name == 'item':
            return self.item_embeddings[ids]
        else:
            return None
        
    def predict_recon(self, user_emb,summary_encoder):
        user_emb = summary_encoder(user_emb)
       
        pred = user_emb @ self.item_embeddings.weight.t()

        return pred




class MatrixFactorizationLLM(nn.Module):
    def __init__(self, num_users, user_embedder,num_items, args):
        super(MatrixFactorizationLLM, self).__init__()

        self.device = args.device
        self.num_users = num_users
        self.num_items = num_items
        self.dim = args.output_emb
        
        
        self.user_embeddings = user_embedder
        self.item_embeddings = nn.Embedding(self.num_items, args.output_emb).to(self.device)
        # self.user_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.user_embeddings.weight.data)
        self.user_embeddings.apply(init_weights)
        self.item_embeddings.weight.data = torch.nn.init.xavier_uniform_(self.item_embeddings.weight.data)

        self.myparameters = [self.user_embeddings.parameters, self.item_embeddings.weight]

    @staticmethod
    def bpr_loss(users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1, keepdim=True)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1, keepdim=True)

        tmp = pos_scores - neg_scores

        bpr_loss = -torch.log( nn.Sigmoid()(tmp) )
        # mf_loss = -torch.sum(maxi)

        return bpr_loss

    def forward(self, user, pos, neg):
        # thr: threshold

        # user_id = torch.from_numpy(user).type(torch.LongTensor).to(self.device)
        pos_id = torch.from_numpy(pos).type(torch.LongTensor).to(self.device)
        neg_id = torch.from_numpy(neg).type(torch.LongTensor).to(self.device)

        user_emb ,pos_emb, neg_emb = \
            self.user_embeddings(user),self.item_embeddings(pos_id), self.item_embeddings(neg_id)

        return  user_emb, pos_emb, neg_emb
    
    
    def get_user_embeddings():
        pass
    def predict(self, user_emb):
        user_emb = self.user_embeddings(user_emb)
       
        pred = user_emb @ self.item_embeddings.weight.t()

        return pred


    def get_embeddings(self, ids, emb_name):
        if emb_name == 'user':
            return self.user_embeddings[ids]
        elif emb_name == 'item':
            return self.item_embeddings[ids]
        else:
            return None
