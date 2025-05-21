import csv
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

def dump_tensor_to_csv(tensor, filepath):
    """Dump a tensor to a CSV file."""
    # Convert tensor to numpy, flattening if needed.
    array = tensor.detach().cpu().numpy()
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        if array.ndim == 1:
            for val in array:
                writer.writerow([val])
        else:
            for row in array:
                writer.writerow(row)

def manual_multihead_attention(q, k, v, module, attn_mask=None, key_padding_mask=None):
    """
    Manually compute multihead attention using module parameters.
    q,k,v: tensors of shape (L, N, E) as passed in (or extracted from tuple if needed).
    module: the nn.MultiheadAttention instance from which we extract parameters.
    Returns the attention output computed manually.
    """
    # Retrieve parameters
    embed_dim = module.embed_dim        # total embedding dimension, E
    nhead = module.num_heads            # number of heads
    head_dim = embed_dim // nhead
    scaling = head_dim ** -0.5

    # Assumes in_proj_weight is concatenation of q, k, v projection weights.
    # Split them into three pieces.
    weight = module.in_proj_weight      # shape (3*E, E)
    bias = module.in_proj_bias          # shape (3*E,) if not None

    weight_q = weight[:embed_dim, :]
    weight_k = weight[embed_dim:2*embed_dim, :]
    weight_v = weight[2*embed_dim:, :]
    bias_q = bias[:embed_dim] if bias is not None else None
    bias_k = bias[embed_dim:2*embed_dim] if bias is not None else None
    bias_v = bias[2*embed_dim:] if bias is not None else None

    # Linear projections for q, k, v
    q_proj = F.linear(q, weight_q, bias_q) * scaling
    k_proj = F.linear(k, weight_k, bias_k)
    v_proj = F.linear(v, weight_v, bias_v)

    # Reshape to separate heads.
    # q_proj: (L, N, embed_dim) -> (N*nhead, L, head_dim)
    def reshape_proj(x):
        L, N, _ = x.size()
        x = x.contiguous().view(L, N, nhead, head_dim)
        x = x.permute(1, 2, 0, 3)  # (N, nhead, L, head_dim)
        return x.reshape(N * nhead, L, head_dim)

    q_proj = reshape_proj(q_proj)
    k_proj = reshape_proj(k_proj)
    v_proj = reshape_proj(v_proj)

    # Compute scaled dot-product attention scores.
    # attn_scores: (N*nhead, L, L)
    attn_scores = torch.bmm(q_proj, k_proj.transpose(1, 2))
    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask
    # Note: key_padding_mask is not processed here explicitly.

    attn_weights = F.softmax(attn_scores, dim=-1)
    # Optionally: apply dropout (if module.dropout > 0)
    if module.dropout > 0:
        attn_weights = F.dropout(attn_weights, p=module.dropout, training=module.training)

    # Compute attention output.
    attn_output = torch.bmm(attn_weights, v_proj)  # (N*nhead, L, head_dim)

    # Revert shape to (L, N, embed_dim)
    def revert_shape(x):
        Nn, L, hd = x.size()
        N = Nn // nhead
        x = x.view(N, nhead, L, hd)
        x = x.permute(2, 0, 1, 3).contiguous()  # (L, N, nhead, head_dim)
        return x.view(L, N, embed_dim)
    attn_output = revert_shape(attn_output)

    # Final projection using out_proj.
    attn_output_manual = F.linear(attn_output, module.out_proj.weight, module.out_proj.bias)
    return attn_output_manual

def hook_fn(name):
    def hook(module, inputs, output):
        dump_dir = f"data_dump/{name}"

        q, k, v = inputs[:3]
        # print(f"Dumping q, k, v to {dump_dir}")
        # print(f"q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}")
        dump_tensor_to_csv(q, os.path.join(dump_dir, "q.csv"))
        dump_tensor_to_csv(k, os.path.join(dump_dir, "k.csv"))
        dump_tensor_to_csv(v, os.path.join(dump_dir, "v.csv"))

        # print(f"Dumping output to {dump_dir}")
        # print(f"output shape: {output[0].shape}")
        dump_tensor_to_csv(output[0], os.path.join(dump_dir, "output.csv"))

        # key_padding_mask = inputs[3]
        # dump_tensor_to_csv(key_padding_mask, os.path.join(dump_dir, "key_padding_mask.csv"))
        # attn_mask = inputs[5]
        # dump_tensor_to_csv(attn_mask, os.path.join(dump_dir, "attn_mask.csv"))

        with torch.no_grad():
            manual_output = manual_multihead_attention(q, k, v, module)
        
        match = torch.allclose(output[0], manual_output)
        print(f"Output match: {match}")
        exit(0)
    return hook

class ACTPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        print(f'KL Weight {self.kl_weight}')

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight
            return loss_dict
        else: # inference time
            a_hat, _, (_, _) = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model # decoder
        self.optimizer = optimizer

    def __call__(self, qpos, image, actions=None, is_pad=None):
        env_state = None # TODO
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None: # training time
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse'] = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else: # inference time
            a_hat = self.model(qpos, image, env_state) # no action, sample from prior
            return a_hat

    def configure_optimizers(self):
        return self.optimizer

def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
