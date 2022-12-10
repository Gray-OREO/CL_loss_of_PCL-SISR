import torch
from pos_neg_examples_generator import get_pos_samples, get_neg_samples


def PixWise_CosSim(im1, im2):
    v1 = torch.flatten(im1, start_dim=1).t()
    v2 = torch.flatten(im2, start_dim=1).t()
    sim = torch.cosine_similarity(v1, v2, dim=1)
    return sum(sim)/v1.shape[0]


def InfoNCE_SingleLayer(querys, positive_keys, negative_keys, temperature=0.1):
    B = querys.shape[0]
    l = 0
    K_p = positive_keys.shape[1]
    K_n = negative_keys.shape[1]
    for b in range(B):
        anchor = querys[b]
        pos_es = positive_keys[b]
        neg_es = negative_keys[b]
        for j in range(K_p):
            p_j = pos_es[j]
            res = PixWise_CosSim(anchor, p_j)
            numerator = torch.exp(res / temperature)
            denominator = torch.clone(numerator)
            for k in range(K_n):
                n_k = neg_es[k]
                denominator += torch.exp(PixWise_CosSim(anchor, n_k) / temperature)
            l += -torch.log(numerator/denominator)
    return l/(K_p*B)


if __name__ == '__main__':
    inp = torch.randn(2, 3, 255, 255)
    n_es = get_neg_samples(inp)  # [b , n, c, h, w]
    p_es = get_pos_samples(inp)
    cl_loss = InfoNCE_SingleLayer(querys=inp, positive_keys=p_es, negative_keys=n_es, temperature=0.4)
    print(cl_loss)