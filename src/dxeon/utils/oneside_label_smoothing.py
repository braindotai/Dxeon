import torch

@torch.no_grad()
def smooth_one_hot(true_labels: torch.Tensor, classes: int, smoothing: float = 0.0):
    '''
    https://github.com/pytorch/pytorch/issues/7455#issuecomment-513735962
    '''
    assert 0 <= smoothing < 1
    
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    
    true_dist = torch.empty(size=label_shape, device=true_labels.device)
    true_dist.fill_(smoothing / (classes - 1))
    true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    
    return true_dist