import torch

import torch.nn as nn


def cosine_similarity(grad1, grad2):
    dot_product = (grad1 * grad2).sum()
    norm_grad1 = grad1.norm()
    norm_grad2 = grad2.norm()
    # return dot_product / (norm_grad1 * norm_grad2)
    return dot_product
