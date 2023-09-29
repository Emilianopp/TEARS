import torch
import math


    
    
def precision_and_recall_k(user_emb, item_emb, train_user_list, test_user_list, klist, args, batch=512):
    """Compute precision at k using GPU.

    Args:
        user_emb (torch.Tensor): embedding for user [user_num, dim]
        item_emb (torch.Tensor): embedding for item [item_num, dim]
        train_user_list (list(set)):
        test_user_list (list(set)):
        k (list(int)):
    Returns:
        (torch.Tensor, torch.Tensor) Precision and recall at k
    """
    # Calculate max k value
    max_k = max(klist)

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.tensor(list(train_user_list[i+j])).to(args.device), value=torch.tensor(0.0).to(args.device))
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=max_k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()
    # Sort indice and get test_pred_topk
    precisions, recalls = [], []
    for k in klist:
        precision, recall = 0, 0
        for i in range(user_emb.shape[0]):
            test = set(test_user_list[i])
            pred = set(result[i, :k].numpy().tolist())
            val = len(test & pred)
            precision += val / max([min([k, len(test)]), 1])
            recall += val / max([len(test), 1])
        precisions.append(precision / user_emb.shape[0])
        recalls.append(recall / user_emb.shape[0])
    return precisions, recalls



def ndcg_k(user_emb, item_emb, train_user_list, test_user_list, k, args, batch=2048):

    # Compute all pair of training and test record

    result = None
    for i in range(0, user_emb.shape[0], batch):
        # print(i)
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            try:
                mask[j].scatter_(dim=0, index=torch.LongTensor(list(train_user_list[i+j])).to(args.device), value=torch.tensor(0.0).to(args.device))
            except:
                pass
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k, dim=1)
        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

    result = result.cpu()
    result = result.numpy().tolist()
    # print(result[666])
    # print(test_user_list[666])


    ndcg = 0
    for user_id in range(len(test_user_list)):
        k = min(k, len(test_user_list[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(result[user_id][j] in set(test_user_list[user_id])) / math.log(j+2, 2) for j in range(k)])
        ndcg += dcg_k / idcg

    return ndcg / float(len(test_user_list))

    # Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res


def build_ndcg_k_list(user_emb, item_emb, train_user_list, test_user_list, topk, batch=200):

    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.LongTensor(list(train_user_list[i+j])).cuda(), value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        # cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=topk, dim=1)

        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
        


    result = result.cpu()
    result = result.numpy().tolist()
    # print(result[4])
    # print(test_user_list[4])

    list_ndcg = []
    ndcg = 0
    for user_id in range(len(test_user_list)):
        k = min(topk, len(test_user_list[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(result[user_id][j] in set(test_user_list[user_id])) / math.log(j+2, 2) for j in range(topk)])
        list_ndcg.append(dcg_k / idcg)

    return list_ndcg

def build_topk_list(user_emb, item_emb, train_user_list, test_user_list, topk, batch=200):
    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.LongTensor(list(train_user_list[i+j])).cuda(), value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        # cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=topk, dim=1)

        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
        


    result = result.cpu()
    result = result.numpy().tolist()
    return result


def build_topk_list_tensor(user_emb, item_emb, train_user_list, test_user_list, topk, batch=200):
    # Compute all pair of training and test record
    result = None
    for i in range(0, user_emb.shape[0], batch):
        # Create already observed mask
        mask = user_emb.new_ones([min([batch, user_emb.shape[0]-i]), item_emb.shape[0]])
        for j in range(batch):
            if i+j >= user_emb.shape[0]:
                break
            mask[j].scatter_(dim=0, index=torch.LongTensor(list(train_user_list[i+j])).cuda(), value=torch.tensor(0.0).cuda())
        # Calculate prediction value
        cur_result = torch.mm(user_emb[i:i+min(batch, user_emb.shape[0]-i), :], item_emb.t())
        # cur_result = torch.sigmoid(cur_result)
        assert not torch.any(torch.isnan(cur_result))
        # Make zero for already observed item
        cur_result = torch.mul(mask, cur_result)
        _, cur_result = torch.topk(cur_result, k=topk, dim=1)

        result = cur_result if result is None else torch.cat((result, cur_result), dim=0)
        result = result.cuda()
        


    return result






