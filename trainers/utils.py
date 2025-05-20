import torch


def recall(scores, labels, k):
    scores = scores
    labels = labels
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hit = labels.gather(1, cut)
    return (hit.sum(1).float() / torch.min(torch.Tensor([k]).to(hit.device), labels.sum(1).float())).mean().cpu().item()


def ndcg(scores, labels, k):
    scores = scores.cpu()
    labels = labels.cpu()
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(2, 2+k)
    weights = 1 / torch.log2(position.float())
    dcg = (hits.float() * weights).sum(1)
    idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in labels.sum(1)])
    ndcg = dcg / idcg
    return ndcg.mean()


def map_at_k(scores, labels, k):
    """
    Calculate Mean Average Precision at k (MAP@k)
    
    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores for each item
    labels : torch.Tensor
        Ground truth labels (1 for relevant items, 0 for non-relevant)
    k : int
        Number of items to consider
    
    Returns
    -------
    float
        Mean Average Precision at k
    """
    scores = scores.cpu()
    labels = labels.cpu()
    
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    hits = labels.gather(1, cut)
    position = torch.arange(1, k + 1)
    precision = hits.cumsum(1) / position
    ap = (precision * hits).sum(1) / torch.min(torch.Tensor([k]).to(hits.device), labels.sum(1).float())
    return ap.mean().item()


def novelty_at_k(scores, labels, k, item_popularity):
    """
    Calculate Novelty@k using Mean Inverse User Frequency (MIUF)
    
    Parameters
    ----------
    scores : torch.Tensor
        Predicted scores for each item
    labels : torch.Tensor
        Ground truth labels (1 for relevant items, 0 for non-relevant)
    k : int
        Number of items to consider
    item_popularity : torch.Tensor
        Number of users who interacted with each item
    
    Returns
    -------
    float
        Novelty@k score
    """
    scores = scores.cpu()
    labels = labels.cpu()
    item_popularity = item_popularity.cpu()
    
    rank = (-scores).argsort(dim=1)
    cut = rank[:, :k]
    n_users = item_popularity.sum()
    item_novelty = -torch.log2(item_popularity / n_users)
    recommended_novelty = item_novelty[cut]
    user_novelty = recommended_novelty.mean(dim=1)
    return user_novelty.mean().item()


def recalls_and_ndcgs_for_ks(scores, labels, ks, item_popularity=None):
    metrics = {}

    scores = scores
    labels = labels
    answer_count = labels.sum(1)

    labels_float = labels.float()
    rank = (-scores).argsort(dim=1)
    cut = rank
    for k in sorted(ks, reverse=True):
       cut = cut[:, :k]
       hits = labels_float.gather(1, cut)
       metrics['Recall@%d' % k] = \
           (hits.sum(1) / torch.min(torch.Tensor([k]).to(labels.device), labels.sum(1).float())).mean().cpu().item()

       position = torch.arange(2, 2+k)
       weights = 1 / torch.log2(position.float())
       dcg = (hits * weights.to(hits.device)).sum(1)
       idcg = torch.Tensor([weights[:min(int(n), k)].sum() for n in answer_count]).to(dcg.device)
       ndcg = (dcg / idcg).mean()
       metrics['NDCG@%d' % k] = ndcg.cpu().item()
       
       # Add MAP@k metric
       metrics['MAP@%d' % k] = map_at_k(scores, labels, k)
       
       # Add Novelty@k metric if item_popularity is provided
       if item_popularity is not None:
           metrics['Novelty@%d' % k] = novelty_at_k(scores, labels, k, item_popularity)

    return metrics