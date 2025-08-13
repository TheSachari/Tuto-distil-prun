import torch.nn as nn
import torch.nn.utils.prune as prune

def apply_global_unstructured_pruning(model: nn.Module, amount: float = 0.5):
    params = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            params.append((m, "weight"))
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    for (m, _) in params:
        prune.remove(m, "weight")
    return model
