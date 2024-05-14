import math
import numpy as np
import bottleneck as bn
import torch



def Recall_at_k_batch(X_pred, heldout_batch, k=100,mean = True):
    batch_users = X_pred.shape[0]
    idx = bn.argpartition(-X_pred, k, axis=1)
    X_pred_binary = np.zeros_like(X_pred, dtype=bool)
    X_pred_binary[np.arange(batch_users)[:, np.newaxis], idx[:, :k]] = True
    X_true_binary = (heldout_batch > 0)
    tmp = (np.logical_and(X_true_binary, X_pred_binary).sum(axis=1)).astype(
        np.float32)
    recall = tmp / np.minimum(k, X_true_binary.sum(axis=1))
    return recall.mean() if mean else recall


def NDCG_binary_at_k_batch(X_pred, heldout_batch, k=100):
    '''
    Normalized Discounted Cumulative Gain@k for binary relevance
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]

    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / np.log2(np.arange(2, k + 2))


    DCG = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                         idx_topk] * tp).sum(axis=1)


    IDCG = np.array([(tp[:min(n, k)]).sum()
                     for n in np.count_nonzero(heldout_batch,axis = 1)])
    # print(f"{IDCG=}")

    return DCG / IDCG

def MRR_at_k(X_pred, heldout_batch, k=100, mean = True):
    '''
    Mean Reciprocal Rank@k
    ASSUMPTIONS: all the 0's in heldout_data indicate 0 relevance
    '''
    batch_users = X_pred.shape[0]

    idx_topk_part = bn.argpartition(-X_pred, k, axis=1)
    topk_part = X_pred[np.arange(batch_users)[:, np.newaxis],
                       idx_topk_part[:, :k]]
    idx_part = np.argsort(-topk_part, axis=1)

    idx_topk = idx_topk_part[np.arange(batch_users)[:, np.newaxis], idx_part]

    tp = 1. / (np.arange(1, k + 1))

    RR = (heldout_batch[np.arange(batch_users)[:, np.newaxis],
                        idx_topk] * tp).max(axis=1)

    return np.mean(RR) if mean else RR


