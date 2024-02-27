
import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, embeddings, positives):
        scores = F.binary_cross_entropy_with_logits(embeddings, positives.float(), reduction='none')
        return scores.mean(dim= 1).mean()

class BinaryCrossEntropyLossSoftmax(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLossSoftmax, self).__init__()

    def forward(self, embeddings, positives):


        BCE = -torch.mean(torch.mean(F.log_softmax(embeddings, 1) * positives, -1))



        return BCE
    



class KLDivergenceLoss(nn.Module):
    def __init__(self,args):
        super(KLDivergenceLoss,self ).__init__()
        self.temp = args.temp
        #setup anneal schedule 


    def forward(self, embeddings, positives):
        

        probs = F.log_softmax(embeddings/self.temp, dim = 1)
        targets = F.softmax(positives.float()/self.temp, dim = 1)

        scores = F.kl_div(probs, targets, reduction='none')

        return scores.sum(dim=1).mean()
def get_loss(loss_name,args=None):
    if loss_name == 'bce':
        return BinaryCrossEntropyLoss()
    elif loss_name == 'bce_softmax':
        return BinaryCrossEntropyLossSoftmax()
    elif loss_name == 'kl':
        return KLDivergenceLoss(args)
    else:
        raise ValueError(f"loss {loss_name} not supported")