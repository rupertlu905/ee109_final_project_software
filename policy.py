import csv
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed

def dump_tensor(tensor, filepath):
    """Dump a tensor to a binary file."""
    # Convert tensor to numpy, flattening if needed.
    array = tensor.detach().cpu().numpy().flatten()
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    with open(filepath, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for element in array:
            writer.writerow([element])

def manual_multihead_attention(q, k, v, module, dump_dir, attn_mask=None, key_padding_mask=None):
    """
    Manually compute multihead attention using module parameters.
    q,k,v: tensors of shape (L, N, E) as passed in (or extracted from tuple if needed).
    module: the nn.MultiheadAttention instance from which we extract parameters.
    Returns the attention output computed manually.
    """
    params_path = os.path.join(dump_dir, "params.txt")
    if not os.path.exists(params_path):
        os.makedirs(dump_dir, exist_ok=True)
    with open(os.path.join(dump_dir, "params.txt"), "w") as f:
        f.write(f"embed_dim: {module.embed_dim}\n")
        f.write(f"num_heads: {module.num_heads}\n")
        f.write(f"q shape: {q.shape}\n")
        f.write(f"k shape: {k.shape}\n")
        f.write(f"v shape: {v.shape}\n")
    dump_tensor(q, os.path.join(dump_dir, "q.csv"))
    dump_tensor(k, os.path.join(dump_dir, "k.csv"))
    dump_tensor(v, os.path.join(dump_dir, "v.csv"))

    # Create a random (2,1,3) matrix called input
    if not os.path.exists("data_dump/test/"):
        os.makedirs("data_dump/test/", exist_ok=True)
    input = torch.randn(2, 1, 3)
    with open(os.path.join("data_dump/test/", "params.txt"), "w") as f:
        f.write(f"input shape: {input.shape}\n")
    dump_tensor(input, os.path.join("data_dump/test/", "input.csv"))

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
    dump_tensor(weight_q, os.path.join(dump_dir, "weight_q.csv"))
    weight_k = weight[embed_dim:2*embed_dim, :]
    dump_tensor(weight_k, os.path.join(dump_dir, "weight_k.csv"))
    weight_v = weight[2*embed_dim:, :]
    dump_tensor(weight_v, os.path.join(dump_dir, "weight_v.csv"))
    bias_q = bias[:embed_dim] if bias is not None else None
    dump_tensor(bias_q, os.path.join(dump_dir, "bias_q.csv"))
    bias_k = bias[embed_dim:2*embed_dim] if bias is not None else None
    dump_tensor(bias_k, os.path.join(dump_dir, "bias_k.csv"))
    bias_v = bias[2*embed_dim:] if bias is not None else None
    dump_tensor(bias_v, os.path.join(dump_dir, "bias_v.csv"))

    with open(os.path.join(dump_dir, "params.txt"), "a") as f:
        f.write(f"weight_q shape: {weight_q.shape}\n")
        f.write(f"weight_k shape: {weight_k.shape}\n")
        f.write(f"weight_v shape: {weight_v.shape}\n")
        f.write(f"bias_q shape: {bias_q.shape if bias_q is not None else None}\n")
        f.write(f"bias_k shape: {bias_k.shape if bias_k is not None else None}\n")
        f.write(f"bias_v shape: {bias_v.shape if bias_v is not None else None}\n")
    
    # Create a random (3, 3) matrix called weight
    weight = torch.randn(3, 3)
    with open(os.path.join("data_dump/test/", "params.txt"), "a") as f:
        f.write(f"weight shape: {weight.shape}\n")
    dump_tensor(weight, os.path.join("data_dump/test/", "weight.csv"))
    # Create a random (3,) vector called bias
    bias = torch.randn(3)
    with open(os.path.join("data_dump/test/", "params.txt"), "a") as f:
        f.write(f"bias shape: {bias.shape}\n")
    dump_tensor(bias, os.path.join("data_dump/test/", "bias.csv"))

    # Linear projections for q, k, v
    q_proj = F.linear(q, weight_q, bias_q) * scaling
    dump_tensor(q_proj, os.path.join(dump_dir, "q_proj.csv"))
    k_proj = F.linear(k, weight_k, bias_k)
    dump_tensor(k_proj, os.path.join(dump_dir, "k_proj.csv"))
    v_proj = F.linear(v, weight_v, bias_v)
    dump_tensor(v_proj, os.path.join(dump_dir, "v_proj.csv"))

    with open(os.path.join(dump_dir, "params.txt"), "a") as f:
        f.write(f"q_proj shape: {q_proj.shape}\n")
        f.write(f"k_proj shape: {k_proj.shape}\n")
        f.write(f"v_proj shape: {v_proj.shape}\n")
    
    output = F.linear(input, weight, bias)
    dump_tensor(output, os.path.join("data_dump/test/", "output.csv"))
    with open(os.path.join("data_dump/test/", "params.txt"), "a") as f:
        f.write(f"output shape: {output.shape}\n")

    ###############################################################################################

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

    if not os.path.exists("data_dump/test2/"):
        os.makedirs("data_dump/test2/", exist_ok=True)
    query = torch.randn(2, 1, 6)
    key = torch.randn(3, 1, 6)
    dump_tensor(query, os.path.join("data_dump/test2/", "query.csv"))
    dump_tensor(key, os.path.join("data_dump/test2/", "key.csv"))
    with open(os.path.join("data_dump/test2/", "params.txt"), "w") as f:
        f.write(f"query shape: {query.shape}\n")
        f.write(f"key shape: {key.shape}\n")
        f.write(f"embed_dim: 6\n")
        f.write(f"n_head: 2\n")
        f.write(f"head_dim: 3\n")
    def reshape_proj_test(x):
        L, N, _ = x.size()
        x = x.contiguous().view(L, N, 2, 3)
        x = x.permute(1, 2, 0, 3)  # (N, nhead, L, head_dim)
        return x.reshape(N * 2, L, 3)
    query = reshape_proj_test(query)
    key = reshape_proj_test(key)

    # Compute scaled dot-product attention scores.
    # attn_scores: (N*nhead, L, L)
    attn_scores = torch.bmm(q_proj, k_proj.transpose(1, 2))
    if attn_mask is not None:
        attn_scores = attn_scores + attn_mask
    # Note: key_padding_mask is not processed here explicitly.

    output = torch.bmm(query, key.transpose(1, 2))
    dump_tensor(output, os.path.join("data_dump/test2/", "attn_scores.csv"))
    with open(os.path.join("data_dump/test2/", "params.txt"), "a") as f:
        f.write(f"attn_scores shape: {output.shape}\n")


    attn_weights = F.softmax(attn_scores, dim=-1)
    output = F.softmax(output, dim=-1)
    dump_tensor(output, os.path.join("data_dump/test2/", "attn_weights.csv"))
    with open(os.path.join("data_dump/test2/", "params.txt"), "a") as f:
        f.write(f"attn_weights shape: {output.shape}\n")

    ###############################################################################################

    # Optionally: apply dropout (if module.dropout > 0)
    if module.dropout > 0:
        attn_weights = F.dropout(attn_weights, p=module.dropout, training=module.training)
    print(f"Dropout: {module.dropout}")

    ###############################################################################################

    # Compute attention output.
    attn_output = torch.bmm(attn_weights, v_proj)  # (N*nhead, L, head_dim)

    if not os.path.exists("data_dump/test3/"):
        os.makedirs("data_dump/test3/", exist_ok=True)
    with open(os.path.join("data_dump/test3/", "params.txt"), "w") as f:
        f.write(f"embed_dim: 6\n")
        f.write(f"n_head: 2\n")
        f.write(f"head_dim: 3\n")
        f.write(f"L_q: 2\n")
        f.write(f"L_k: 3\n")
        f.write(f"L_v: 3\n")
    attn_weights = torch.randn(2, 2, 3)
    value = torch.randn(3, 1, 6)
    dump_tensor(attn_weights, os.path.join("data_dump/test3/", "attn_weights.csv"))
    dump_tensor(value, os.path.join("data_dump/test3/", "value.csv"))
    with open(os.path.join("data_dump/test3/", "params.txt"), "a") as f:
        f.write(f"attn_weights shape: {attn_weights.shape}\n")
        f.write(f"value shape: {value.shape}\n")

    # Revert shape to (L, N, embed_dim)
    def revert_shape(x):
        Nn, L, hd = x.size()
        N = Nn // nhead
        x = x.view(N, nhead, L, hd)
        x = x.permute(2, 0, 1, 3).contiguous()  # (L, N, nhead, head_dim)
        return x.view(L, N, embed_dim)
    attn_output = revert_shape(attn_output)

    def revert_shape_test(x):
        Nn, L, hd = x.size()
        N = Nn // 2
        x = x.view(N, 2, L, hd)
        x = x.permute(2, 0, 1, 3).contiguous()  # (L, N, nhead, head_dim)
        return x.view(L, N, 6)
    value = reshape_proj_test(value)
    output = torch.bmm(attn_weights, value)
    output = revert_shape_test(output)
    dump_tensor(output, os.path.join("data_dump/test3/", "attn_output.csv"))
    with open(os.path.join("data_dump/test3/", "params.txt"), "a") as f:
        f.write(f"attn_output shape: {output.shape}\n")

    ###############################################################################################

    # Final projection using out_proj.
    attn_output_manual = F.linear(attn_output, module.out_proj.weight, module.out_proj.bias)

    if not os.path.exists("data_dump/test4/"):
        os.makedirs("data_dump/test4/", exist_ok=True)
    with open(os.path.join("data_dump/test4/", "params.txt"), "w") as f:
        f.write(f"embed_dim: 6\n")
        f.write(f"L_q: 2\n")
    attn_output = torch.randn(2, 1, 6)
    weight = torch.randn(6, 6)
    bias = torch.randn(6)
    dump_tensor(attn_output, os.path.join("data_dump/test4/", "attn_output.csv"))
    dump_tensor(weight, os.path.join("data_dump/test4/", "weight.csv"))
    dump_tensor(bias, os.path.join("data_dump/test4/", "bias.csv"))
    with open(os.path.join("data_dump/test4/", "params.txt"), "a") as f:
        f.write(f"attn_output shape: {attn_output.shape}\n")
        f.write(f"weight shape: {weight.shape}\n")
        f.write(f"bias shape: {bias.shape}\n")
    final_output = F.linear(attn_output, weight, bias)
    dump_tensor(final_output, os.path.join("data_dump/test4/", "final_output.csv"))
    with open(os.path.join("data_dump/test4/", "params.txt"), "a") as f:
        f.write(f"final_output shape: {final_output.shape}\n")

    return attn_output_manual

def hook_fn(name):
    def hook(module, inputs, output):
        dump_dir = f"data_dump/{name}"

        q, k, v = inputs[:3]

        with torch.no_grad():
            manual_output = manual_multihead_attention(q, k, v, module, dump_dir)
        
        match = torch.allclose(output[0], manual_output)
        print(f"Output match: {match}")
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
