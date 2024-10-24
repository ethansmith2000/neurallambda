'''.

Continue the t13 series, and do N-way k-shot learning task, where a transformer
backbone outputs LOR updates to itself. Trained with "train-time-data", so
hopefully, at inference we won't even need metalearning!

----------
Model/Data Design

- N*k support images embedded continuously as inputs
- N*k labels, tokenized and embedded, as standard
- N*q query images
- N*q query labels
- After all support images, randn LoR tokens are appended
- Things I might do:
    - These tokens get trained via metalearning at train time
    - The resulting lors from metalearning could be trained as targets for the outer loop ("train-time-data")
    - Do we need metalearning at inference time?

----------
N-way k-shot learning task:
    1. N classes: The task tests classification between N different classes that the model has not seen during training.
    2. k examples per class: For each of the N classes, the model is given only k labeled examples (usually k is a small number like 1 or 5).
    3. Support set: The N*k labeled examples make up the "support set" that the model can use to learn about the new classes.
    4. Query set: The model is then asked to classify new unlabeled examples (the "query set") from the N classes.
    5. Meta-learning: Models are typically trained on many different N-way k-shot tasks so they can learn to adapt quickly to new tasks at test time.
    6. Evaluation: Performance is measured by classification accuracy on the query set.

Metalearning with pretraining on Omniglot, Testing on Mini-Imagenet Test


PROVENANCE:
- t13_metalearning_hypernet_03


NaN ISSUES:
- rmsprop and adam have issues during the backward pass of the outerloop, and throw NaNs (at least I think this is where the NaNs are coming from. NaNs show up in weights after outer loop, so immediately bork the next fwd pass)


SPEEDING UP ITERATION:
- truncating model
- smaller N classes helps speed up iteration
- turn off tensorboard logging


STABILIZATION:
- sgd_momentum, lr=1.0, N_LOOP=20
- rmsprop and adam NaN issue
- rm Dropout in LORProject, helped a ton (DID IT?)
- randn up-down random intermediate seems important. Without, i can reach "better than chance" but even that's unstable. Using a projection instead of randn seems ok; starts off way worse so init matters, and is less stable.
- LORNorm: small init (normal_init * 1e-2) for RMSNorm helps a lot

DESIGN CHOICES:
- MLPMixer (rm dropout. intermediate size?. residuals? act fn? RMSNorm?)
- randn ud_intermediate in MLPMixer
- randn kv_intermediate in MLPMixer
- hyperparams (lr, wd, frozen params)
- metatokens as randn?  (embeddings seem to help, maybe)
- feedback updated metaweights each inner loop?
- img_proj: architecture and init

TODO:
- [ ] fix run_epoch
  - [X] update for new dataset
  - [X] add image projector
  - [X] tokenize labels
  - [X] append metatokens
  - [X] metalearn them
  - [X] prove loss can decrease for supports before worrying about queries
  - [ ] test on queries

- options
  1. [ ] keep model frozen, only train LORProject
  2. [ ] use metalearned tokens in outerloop, and train entire model

'''


import os
import sys
import traceback
import importlib
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from functools import partial
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT
from neurallambda.lab.common import print_model_info

if 't14_homoiconic_llm_model_03' in sys.modules:
    print('RELOADING Q')
    importlib.reload(sys.modules['t14_homoiconic_llm_model_03'])
else:
    import t14_homoiconic_llm_model_03 as Q


current_time = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
writer = SummaryWriter(f'log/t14_homoiconic_llm_test_metalearning_2/{current_time}')

LOG = False  # slows down training to ~0.5x

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'
WHICH_LOR = 1
N_METAWEIGHTS = 14  # QKVOGUD * 2
LOR_LAYER = 2  # target of LOR stuff


##################################################
# Load Model

model_name = os.path.expanduser("~/_/models/Qwen2-0.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-1.5B")
# model_name = os.path.expanduser("~/_/models/Qwen2-7B")

def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        isnan = torch.isnan(output).any()
        isinf = torch.isinf(output).any()
        if isnan or isinf:
            print('\n' * 3)
            print(f"NaN or inf detected in {module.__class__.__name__}: isnan:{isnan.item()}, isinf{isinf.item()}")
            print(f"Module: {module}")
            print(f"Module name: {module_names.get(module, 'Unknown')}")
            print(f"Input shape: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
            print(f"Output shape: {output.shape}")
            breakpoint()
            # raise RuntimeError("NaN or inf detected: {isnan=}, {isinf=}")


def hook_fn(module, input, output):
    if isinstance(output, torch.Tensor):
        # isnan_in = torch.isnan(input).any()  # inputs are a tuple
        # isinf_in = torch.isinf(input).any()
        isnan_out = torch.isnan(output).any()
        isinf_out = torch.isinf(output).any()
        if isnan_out or isinf_out:  # or isnan_in or isinf_in:
            # Get the current stack trace
            stack = traceback.extract_stack()
            # Remove the last entry which is this function call
            stack = stack[:-1]

            print('\n' * 3)
            print("=" * 80)
            print("NaN/Inf Detection Report")
            print("=" * 80)

            print("\nISSUE DETECTED:")
            # print(f"- NaN Input detected: {isnan_in.item()}")
            # print(f"- Inf Input detected: {isinf_in.item()}")
            print(f"- NaN Output detected: {isnan_out.item()}")
            print(f"- Inf Output detected: {isinf_out.item()}")

            print("\nMODULE INFORMATION:")
            print(f"- Class: {module.__class__.__name__}")
            print(f"- Full module: {module}")
            print(f"- Module name: {module_names.get(module, 'Unknown')}")

            print("\nTENSOR SHAPES:")
            print(f"- Input shapes: {[i.shape if isinstance(i, torch.Tensor) else type(i) for i in input]}")
            print(f"- Output shape: {output.shape}")

            print("\nSTACK TRACE:")
            for filename, lineno, funcname, line in stack:
                rel_path = os.path.relpath(filename)  # Get relative path for cleaner output
                if 'site-packages' in rel_path:  # skip library stack trace
                    continue
                print(f"  File '{rel_path}', line {lineno}, in {funcname}")
                if line:
                    print(f"    {line.strip()}")

            print("\nTENSOR STATISTICS:")

            # if not torch.all(torch.isinf(input)):
            #     print("- Input stats (excluding inf):")
            #     valid_input = input[~torch.isinf(input)]
            #     if len(valid_input) > 0:
            #         print(f"  - Mean: {valid_input.mean().item():.6f}")
            #         print(f"  - Std: {valid_input.std().item():.6f}")
            #         print(f"  - Min: {valid_input.min().item():.6f}")
            #         print(f"  - Max: {valid_input.max().item():.6f}")

            if not torch.all(torch.isinf(output)):
                print("- Output stats (excluding inf):")
                valid_output = output[~torch.isinf(output)]
                if len(valid_output) > 0:
                    print(f"  - Mean: {valid_output.mean().item():.6f}")
                    print(f"  - Std: {valid_output.std().item():.6f}")
                    print(f"  - Min: {valid_output.min().item():.6f}")
                    print(f"  - Max: {valid_output.max().item():.6f}")

            print("\nDebugger starting...")
            breakpoint()

module_names = {}

def add_hooks(model):
    for name, module in model.named_modules():
        module_names[module] = name
        module.register_forward_hook(hook_fn)

try:
    fail
    already_loaded
except:
    print('Loading model')
    model = Q.Qwen2ForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.bfloat16,  # HERE BE DRAGONS
        torch_dtype=torch.float32,
        # torch_dtype=torch.float64,
        device_map=DEVICE,
        _attn_implementation='eager',
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'


    add_hooks(model)
    already_loaded = True


if True:
    warnings.warn('TRUNCATING MODEL LAYERS')
    model.model.layers = model.model.layers[0:3] + model.model.layers[-3:]
    for ix in range(len(model.model.layers)):
        model.model.layers[ix].layer_idx = ix
        model.model.layers[ix].self_attn.layer_idx = ix

if True:
    warnings.warn('TRUNCATING MODEL EMBEDDINGS')

    assert id(model.lm_head.weight) == id(model.model.embed_tokens.weight), 'model does not have tied weights, but truncation expects it'

    TRUNCATE_EMB_SIZE = 1000

    new_embs = nn.Parameter(
        model.model.embed_tokens.weight[:TRUNCATE_EMB_SIZE].clone(),
        requires_grad=model.model.embed_tokens.weight.requires_grad
    )

    model.model.embed_tokens.weight = new_embs
    model.model.config.num_hidden_layers = len(model.model.layers)
    model.model.config.vocab_size = model.model.embed_tokens.weight.shape[0]

    model.lm_head.weight = new_embs

num_layers = model.config.num_hidden_layers

print_model_info(model)


##################################################
# Functional Optimizers
#
#   Differentiable variants of: SGD, SGD-Momentum, RMSProp, Adam


##########
# SGD

def sgd(params, grads, lr=0.01):
    return [p - g * lr for p, g in zip(params, grads)]


##########
# SGD Momentum

def sgd_momentum_init(params):
    return [torch.zeros_like(p) for p in params]

def sgd_momentum(params, grads, velocity, lr=0.01, momentum=0.9):
    updated_velocity = [momentum * v + lr * g for v, g in zip(velocity, grads)]
    updated_params = [p - v for p, v in zip(params, updated_velocity)]
    return updated_params, updated_velocity


##########
# RMSProp

def rmsprop_init(params):
    return [torch.zeros_like(p) for p in params]  # square averages

def rmsprop(params, grads, square_avg, lr=0.01, alpha=0.99, eps=1e-8):
    updated_square_avg = [alpha * avg + (1 - alpha) * g.pow(2) for avg, g in zip(square_avg, grads)]
    updated_params = [p - lr * g / (avg.sqrt() + eps)
                      for p, g, avg in zip(params, grads, updated_square_avg)]
    return updated_params, updated_square_avg


##########
# ADAM

def adam_init(params):
    return (
        [torch.zeros_like(p) for p in params],  # m
        [torch.zeros_like(p) for p in params],  # v
        0  # t
    )

def adam(params, grads, m, v, t, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    t += 1
    updated_m = [betas[0] * m_i + (1 - betas[0]) * g
                 for m_i, g in zip(m, grads)]
    updated_v = [betas[1] * v_i + (1 - betas[1]) * g.pow(2)
                 for v_i, g in zip(v, grads)]

    m_hat = [m_i / (1 - betas[0]**t) for m_i in updated_m]
    v_hat = [v_i / (1 - betas[1]**t) for v_i in updated_v]

    updated_params = [p - lr * mh / (vh.sqrt() + eps)
                      for p, mh, vh in zip(params, m_hat, v_hat)]

    return updated_params, updated_m, updated_v, t


##########
# Adamax

def adamax_init(params):
    return (
        [torch.zeros_like(p) for p in params],  # m (first moment)
        [torch.zeros_like(p) for p in params],  # u (infinity norm)
        0  # t (timestep)
    )

def adamax(params, grads, m, u, t, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
    """
    Implements Adamax optimization algorithm (a variant of Adam based on infinity norm).

    Args:
        params: List of parameters
        grads: List of gradients
        m: List of first moment vectors
        u: List of infinity norm vectors
        t: Integer, timestep counter
        lr: Float, learning rate (default: 0.002)
        betas: Tuple of coefficients for computing running averages
        eps: Term for numerical stability

    Returns:
        (updated_params, updated_m, updated_u, t)
    """
    t += 1

    # Update biased first moment estimate
    updated_m = [betas[0] * m_i + (1 - betas[0]) * g
                 for m_i, g in zip(m, grads)]

    # Update the infinity norm estimate
    # u_t = max(beta_2 * u_{t-1}, |g_t|)
    updated_u = [torch.maximum(betas[1] * u_i, torch.abs(g))
                 for u_i, g in zip(u, grads)]

    # Bias correction for the first moment
    # Note: infinity norm doesn't need bias correction
    m_hat = [m_i / (1 - betas[0]**t) for m_i in updated_m]

    # Update parameters
    # theta_t = theta_{t-1} - (alpha / u_t) * m_hat_t
    updated_params = [p - (lr / (u_i + eps)) * mh
                      for p, mh, u_i in zip(params, m_hat, updated_u)]

    return updated_params, updated_m, updated_u, t


##################################################
# LOR stuff


def empty_lors(num_layers):
    '''per-layer cache of all lor blocks parsed during generation, and used in
future generation. The None values at a given layer index will be replaced with
a tuple of tensors shaped like ([BATCH, DIM, N], [BATCH, N, DIM]). N is the
rank of the LOR matrix. In the current implementation, these are just
concatenated, and never erased (within a batch, ofc). '''
    lors = {
        # low rank attention params
        "lor_qs": [None] * num_layers,
        "lor_ks": [None] * num_layers,
        "lor_vs": [None] * num_layers,
        "lor_os": [None] * num_layers,

        # low rank mlp params
        "lor_gs": [None] * num_layers,
        "lor_us": [None] * num_layers,
        "lor_ds": [None] * num_layers,
    }
    return lors


def partially_apply_models(lor_models, lor_cache):
    '''deep in the transformer stack, the LORModules get applied, but they need to
reference the current lor cache. This is where they get it from.'''
    fs = {}
    lor_ix_keys = lor_cache.keys()
    for k in lor_ix_keys:
        fs[k] = []
        for m, c in zip(lor_models[k], lor_cache[k]):
            if m is not None:
                f = partial(m, c)
            else:
                f = None
            fs[k].append(f)
    return fs


def update_lors(
        lor_models,  # shaped like `empty_lors`. Contains lor_proj, and per-head models
        lor_cache,  # shaped like `empty_lors`. Contains previously parsed lors
        lor_ixs_per_layer: List[torch.Tensor],  # List[spans], where a span is [batch, (start_ix, end_ix)]
        hidden_states,  # from out.hidden_states, so, contains all layers
        num_layers,
):
    ''' Update the LORs by interpreting hidden states as new lor blocks '''

    # check that lor_models and lor_cache are defined (at least None values)
    # for each layer, and that there's a model for every cache key
    lor_keys = lor_cache.keys()
    for k in lor_keys:
        assert (
            len(lor_models[k]) ==  # one (optional) lor module per layer
            len(lor_cache[k]) ==  # cache per layer
            num_layers
        ), (f'''
{len(lor_models[k])=} ==  # one (optional) lor module per layer
{len(lor_cache[k])=} ==  # cache per layer
{num_layers=}
''')

    h_emb = hidden_states[-1]  # final layer states

    # iterate over all layers
    for layer_ix in range(num_layers):
        lor_ix_spans = lor_ixs_per_layer[layer_ix]

        # skip non-lor'd layers
        if lor_models['lor_proj'][layer_ix] is None:
            assert lor_ix_spans is None, f'lor_proj is not defined for layer {layer_ix}, but there are lor_ix_spans is defined for this layer'
            continue

        # check that spans are within bounds
        assert isinstance(lor_ix_spans, torch.Tensor)
        assert lor_ix_spans.min() >= -1
        assert lor_ix_spans.max() <= hidden_states[-1].shape[1]

        parses = select_spans(h_emb, lor_ix_spans)

        # no parses implied anywhere
        if (parses > -1).sum() == 0:
            continue

        # run lor_proj. Returns tuple of L and R singular values, per key, eg: (lor_qs_l, lor_qs_r, ...)
        projs = lor_models['lor_proj'][layer_ix](parses)
        proj_pairs = zip(projs[::2], projs[1::2])

        # update cache
        for k, (l, r) in zip(lor_keys, proj_pairs):
            if lor_cache[k][layer_ix] is None:  # is first pass, no cache yet
                lor_cache[k][layer_ix] = (l.unsqueeze(2), r.unsqueeze(2))  # [B, DIM, RANK]
            else:
                lor_cache[k][layer_ix] = (torch.cat([lor_cache[k][layer_ix][0], l.unsqueeze(2)], dim=2),
                                          torch.cat([lor_cache[k][layer_ix][1], r.unsqueeze(2)], dim=2))  # [B, DIM, RANK]

    return lor_cache


def select_spans(x, indices):
    """Selects spans from a 3D tensor (`[batch, seq, dim]`) along dim=1 using
    provided start and end indices.

    Perform span selection on a 3D tensor. If `indices` contains [-1, -1] for a
    batch, that location will be filled with 0s.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seq, dim] where:
            batch: number of sequences in the batch
            seq: length of each sequence
            dim: dimensionality of each token representation

        indices (torch.Tensor): 2D tensor of indices for span selection, with shape
            [batch, 2]. Each row contains [start, end] indices for the span.
            Start and end are inclusive. If a row is [-1, -1], the corresponding
            output will be filled with 0s.

    Returns:
        torch.Tensor: Output tensor of shape [batch, max_span_length, dim]

    Example:
        >>> x = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]],
        ...                   [[13, 14, 15], [16, 17, 18], [19, 20, 21], [22, 23, 24]]])
        >>> indices = torch.tensor([[1, 2], [-1, -1]])
        >>> select_spans(x, indices)
        tensor([[[ 4,  5,  6],
                 [ 7,  8,  9],
                 [ 0,  0,  0]],
                [[ 0,  0,  0],
                 [ 0,  0,  0],
                 [ 0,  0,  0]]])
    """
    B, S, D = x.shape  # batch, sequence length, dimension

    # Create a mask for valid spans (not [-1, -1])
    mask = (indices[:, 0] != -1)  # we assume -1s are paired correctly with another -1

    # Calculate span lengths
    span_lengths = torch.where(mask, indices[:, 1] - indices[:, 0] + 1, torch.zeros_like(indices[:, 0]))
    max_span_length = span_lengths.max().item()

    # Create position indices for each element in the max span
    positions = torch.arange(max_span_length, device=x.device).unsqueeze(0).expand(B, -1)

    # Calculate absolute indices for each position
    abs_indices = indices[:, 0].unsqueeze(1) + positions

    # Create a mask for valid positions within each span
    valid_positions = positions < span_lengths.unsqueeze(1)

    # Combine the span mask and position mask
    final_mask = mask.unsqueeze(1) & valid_positions

    # Create batch indices
    batch_indices = torch.arange(B, device=x.device).unsqueeze(1).expand(-1, max_span_length)

    # Gather values using absolute indices, with out-of-bounds handling
    gathered_values = torch.zeros((B, max_span_length, D), device=x.device, dtype=x.dtype)
    valid_abs_indices = abs_indices[final_mask]
    valid_batch_indices = batch_indices[final_mask]
    gathered_values[final_mask] = x[valid_batch_indices, valid_abs_indices]

    return gathered_values



##################################################
# LOR Models: LORProject + LORNorm


def assert_no_biases(model):
    '''Because of how batching interacts with parsing LoR weights, the LORModule
must not have biases. See LORModule for more details.'''
    bias_info = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            bias_info.append(f"Linear bias found in {name}")

        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            bias_info.append(f"Convolutional bias found in {name}")

        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and module.bias is not None:
            bias_info.append(f"BatchNorm bias found in {name}")

        elif isinstance(module, nn.LayerNorm) and module.bias is not None:
            bias_info.append(f"LayerNorm bias found in {name}")

        elif isinstance(module, (nn.LSTM, nn.GRU)):
            for param_name, param in module.named_parameters():
                if 'bias' in param_name:
                    bias_info.append(f"{type(module).__name__} bias found in {name}.{param_name}")

        elif isinstance(module, nn.Embedding) and module.padding_idx is not None:
            bias_info.append(f"Embedding bias (padding_idx) found in {name}")

    if bias_info:
        error_message = "The model contains biases:\n" + "\n".join(f"- {info}" for info in bias_info)
        raise AssertionError(error_message)


def apply_lor(x, lorl, lorr) -> torch.Tensor:
    ''' Low rank "matrix" multiplication

    args:
      x: [B, S, D]
      lorl: [B, rank, out_features]
      lorr: [B, in_features, rank]

    '''
    x = torch.einsum('bsd, bdr -> bsr', x, lorr)
    x = torch.einsum('bsr, bdr -> bsd', x, lorl)
    return x


class LORProject(nn.Module):
    '''
    LOR weights need to be projected to fit the shapes of the underlying
    Linear layers they're matching. These LORModules solve this, and there
    are 2 modules per target linear layer, (left singular values, right
    singular values). They multiply like: out=LRx, to match the usual out=Wx,
    so the new calculation becomes out=Wx + LRx.

    R are the input vectors, and L are the output vectors. The first
    dimension of the LORModules must match the embedding that we're
    projecting, so the 1st values are all `dim`. The 2nd dim of R is the
    input dimension of the matched linear weights. The 2nd dim of L is the
    output dimension of the same linear layer.

    '''

    def __init__(self, dropout_rate=0.15):
        super().__init__()

        dim = model.model.embed_tokens.weight.shape[1]  # embedding dimension
        k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
        v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
        ff_dim = model.config.intermediate_size

        self.dim = dim
        self.k_dim = k_dim
        self.v_dim = v_dim
        self.ff_dim = ff_dim

        self.input_dims = [dim] * N_METAWEIGHTS  # All inputs are 'dim'-dimensional

        n_out = 14
        self.output_dims = [
            dim, dim,  # lor_qs_l, lor_qs_r
            k_dim, dim,  # lor_ks_l, lor_ks_r
            v_dim, dim,  # lor_vs_l, lor_vs_r
            dim, dim,  # lor_os_l, lor_os_r
            ff_dim, dim,  # lor_gs_l, lor_gs_r
            ff_dim, dim,  # lor_us_l, lor_us_r
            dim, ff_dim  # lor_ds_l, lor_ds_r
        ]

        self.token_mixing_dim = 128
        self.channel_mixing_dim = 128

        # Token-mixing MLP
        self.token_mixing_mlp = nn.Sequential(
            nn.RMSNorm(N_METAWEIGHTS),
            nn.Linear(N_METAWEIGHTS, self.token_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.token_mixing_dim, n_out, bias=False),
            nn.Dropout(dropout_rate),
        )

        # Channel-mixing MLP (same for all inputs)
        self.channel_mixing_mlp = nn.Sequential(
            nn.RMSNorm(dim),
            nn.Linear(dim, self.channel_mixing_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.channel_mixing_dim, dim, bias=False),
            nn.Dropout(dropout_rate),
        )

        # Final projection layers
        self.final_projections = nn.ModuleDict({
            'lor_qs_l': nn.Linear(dim, dim, bias=False),
            'lor_qs_r': nn.Linear(dim, dim, bias=False),
            'lor_ks_l': nn.Linear(dim, k_dim, bias=False),
            'lor_ks_r': nn.Linear(dim, dim, bias=False),
            'lor_vs_l': nn.Linear(dim, v_dim, bias=False),
            'lor_vs_r': nn.Linear(dim, dim, bias=False),
            'lor_os_l': nn.Linear(dim, dim, bias=False),
            'lor_os_r': nn.Linear(dim, dim, bias=False),
            'lor_gs_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_gs_r': nn.Linear(dim, dim, bias=False),
            # 'lor_us_l': nn.Linear(dim, ff_dim, bias=False),
            'lor_us_r': nn.Linear(dim, dim, bias=False),
            'lor_ds_l': nn.Linear(dim, dim, bias=False),
            # 'lor_ds_r': nn.Linear(dim, ff_dim, bias=False),

            # 'lor_us_l_ds_r': nn.Linear(dim, ff_dim, bias=False),

        })


        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)


    def forward(self, x):
        '''
        x: [B, N_METAWEIGHTS, D]
        '''

        B = x.shape[0]
        device = x.device

        # Token-mixing
        # residual = x

        # x_token shape: [batch, dim, 14]
        x_token = x.transpose(1, 2)
        # Normalize across token dimension
        x_token = self.token_mixing_mlp(x_token)
        x_token = x_token.transpose(1, 2)  # [batch, 14, dim]

        # x = residual + x_token

        # Channel-mixing
        # residual = x
        x = self.channel_mixing_mlp(x)

        # x = residual + x

        ##########
        # Tie intermediates. Adopting results from (ie, results from t14_homoiconic_llm_adding_data_to_mlp)

        # # Generated `ud_intermediate`
        # ud_intermediate = self.final_projections['lor_us_l_ds_r'](x[:, 10])

        # # NOTE: statistics of randn are likely way off
        # # TODO: bc of this shrink, shrink token_mixing_mlp from dim=14 to dim=12, since those outputs go unused
        # ud_intermediate = torch.randn(B, self.ff_dim, device=device)
        # outputs = (
        #     self.final_projections['lor_qs_l'](x[:, 0, :]),
        #     self.final_projections['lor_qs_r'](x[:, 1, :]),
        #     self.final_projections['lor_ks_l'](x[:, 2, :]),
        #     self.final_projections['lor_ks_r'](x[:, 3, :]),
        #     self.final_projections['lor_vs_l'](x[:, 4, :]),
        #     self.final_projections['lor_vs_r'](x[:, 5, :]),
        #     self.final_projections['lor_os_l'](x[:, 6, :]),
        #     self.final_projections['lor_os_r'](x[:, 7, :]),
        #     self.final_projections['lor_gs_l'](x[:, 8, :]),
        #     # ud_intermediate * -1,
        #     self.final_projections['lor_gs_r'](x[:, 9, :]),

        #     # self.final_projections['lor_us_l'](x[:, 10, :]),
        #     ud_intermediate,  # replace lor_us_l

        #     self.final_projections['lor_us_r'](x[:, 11, :]),
        #     self.final_projections['lor_ds_l'](x[:, 12, :]),

        #     # self.final_projections['lor_ds_r'](x[:, 13, :]),
        #     ud_intermediate,  # replace lor_ds_r
        # )


        ud_intermediate = torch.randn(B, self.ff_dim, device=device)
        kv_intermediate = torch.randn(B, self.k_dim, device=device)
        ou_intermediate = torch.randn(B, self.dim, device=device)

        outputs = (
            # self.final_projections['lor_qs_l'](x[:, 0, :]),
            torch.cat([kv_intermediate] * (self.dim // self.k_dim), dim=-1),

            self.final_projections['lor_qs_r'](x[:, 1, :]),

            # self.final_projections['lor_ks_l'](x[:, 2, :]),
            kv_intermediate,

            self.final_projections['lor_ks_r'](x[:, 3, :]),

            # self.final_projections['lor_vs_l'](x[:, 4, :]),
            kv_intermediate,

            self.final_projections['lor_vs_r'](x[:, 5, :]),

            # self.final_projections['lor_os_l'](x[:, 6, :]),
            ou_intermediate,

            self.final_projections['lor_os_r'](x[:, 7, :]),

            self.final_projections['lor_gs_l'](x[:, 8, :]),
            # ud_intermediate * -1,

            self.final_projections['lor_gs_r'](x[:, 9, :]),
            # ou_intermediate,

            # self.final_projections['lor_us_l'](x[:, 10, :]),
            ud_intermediate,  # replace lor_us_l

            # self.final_projections['lor_us_r'](x[:, 11, :]),
            ou_intermediate,

            self.final_projections['lor_ds_l'](x[:, 12, :]),

            # self.final_projections['lor_ds_r'](x[:, 13, :]),
            ud_intermediate,  # replace lor_ds_r
        )



        return outputs







class LORNorm(nn.Module):
    def __init__(self, in_dim, out_dim, name):
        super().__init__()

        self.name = name

        self.norm = nn.RMSNorm(out_dim)

        # NOTE: there is probably a principled way of setting this, or
        #   pre-learning this. As it stands, the norm can be initialized even
        #   to 0 which effectively removes this entire layer from the
        #   transformer stack EXCEPT residuals still pass forward. Strangely
        #   enough, the network can do ok with a layer removed. I'm thinking
        #   here that we can limit the initial impact of LoR stuff by setting
        #   this low. This has a SIDE-EFFECT though of small grads to this layer.
        with torch.no_grad():
            self.norm.weight[:] = self.norm.weight * 1e-1

        # This is akin to He initialization. TODO: worth it?
        self.scale = (2 / in_dim) ** 0.5

        # LORModule must play nicely in a batch situation, where some samples
        # of the batch imply lor parses and others don't. Non LoR'd samples
        # should not be affected by sharing a batch with LoR'd samples. Biases
        # corrupt this property. 0-valued lors (from samples without lor
        # parses) must produce 0-valued outputs here. Checking for biases is
        # not the whole solution, you must take care.
        #
        # This is not necessarily necessary. For instance, clever masking of
        # non-parsed samples might obviate this.
        assert_no_biases(self)

    def forward(self, lor_cache, original, hidden_state):
        '''This gets applied separately per QKVOGUD, and really just allows
        Normalization to be handled from this module, instead of say adding a
        new norm layer throughout the underlying model.

        The `project` function is where more magic happens; it takes in ALL QKVOGUD parses for a layer, and generates a cache for each together.

        Args:
          original: model's original values of eg QKVOGUD within this layer
          hidden_state: hidden_state at this layer, that projects through this layer's associated linear QKVOGUD block, and will be used to project through the LoR version too.

        '''
        if lor_cache is not None:
            lorl, lorr = lor_cache
            # print(f'name: {self.name}') # DEBUG
            l = apply_lor(hidden_state, lorl, lorr)
            # return self.norm(original + l * self.scale)  # TODO: revisit if `scale` is good
            return self.norm(original + l)
        else:
            return self.norm(original)



##################################################
# Training


def run_epoch(model, img_proj, lor_models, inner_lr, dataloader, optimizer, device, train=True, debug=False):
    model.train() if train else model.eval()
    total_loss = 0
    total_samples = 0

    num_layers = model.config.num_hidden_layers
    D = model.config.hidden_size
    vocab_size = model.model.embed_tokens.weight.shape[0]

    with torch.set_grad_enabled(train):
        for batch in tqdm(dataloader, desc="Training" if train else "Evaluating"):

            # batch (Tuple[List[Tuple[torch.Tensor, torch.Tensor]], List[Tuple[torch.Tensor, torch.Tensor]]]):
            #     A batch containing a single task, where:
            #     - The first element is the support set: a list of N*k tuples, each containing
            #       (batched_image, batched_label) for support examples.
            #     - The second element is the query set: a list of N*q tuples, each containing
            #       (batched_image, batched_label) for query examples.

            # (Pdb) type(batch)
            # <class 'list'>
            # (Pdb) len(batch)
            # 2
            # (Pdb) type(batch[0])
            # <class 'list'>
            # (Pdb) len(batch[0])
            # 5
            # (Pdb) type(batch[0][0])
            # <class 'list'>
            # (Pdb) len(batch[0][0])
            # 2
            # (Pdb) batch[0][0][0].shape
            # torch.Size([32, 28, 28])
            # (Pdb) batch[0][0][1].shape
            # torch.Size([32])


            # supports: N*k tuples of batched images and labels
            # queries: N tuples (or N*q if multiple queries) of batched images and labels
            supports, queries = batch

            # Move to device, flatten and project images into embedding dim
            support_imgs = img_proj(torch.stack([x[0].to(device).flatten(start_dim=1, end_dim=2) for x in supports], dim=1))  # N*k tensors [B, IMG, IMG] -> [B, N*k, D]
            support_labels = torch.stack([x[1].to(device) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
            query_imgs = img_proj(torch.stack([x[0].to(device).flatten(start_dim=1, end_dim=2) for x in queries], dim=1))  # N*k tensors [B, IMG, IMG] -> [B, N*k, D]
            query_labels = torch.stack([x[1].to(device) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]

            B, Ss = support_labels.shape  # batch size, sequence (N*k)
            Sq = query_labels.shape[1]  # N*q

            # EXPERIMENT: random metaweights
            metaweights = torch.randn(B, N_METAWEIGHTS, D, device=device) * 1e-3  # TODO: what should this init be?

            # # EXPERIMENT: tokens for metaweights
            # metatokens = torch.tensor(list(range(50, 64)), device=device, dtype=torch.long)
            # metaweights = model.model.embed_tokens(metatokens).unsqueeze(0).repeat(B, 1, 1)

            img_attention_mask = torch.ones((B, Ss), device=device, dtype=torch.long)
            img_uncausal_mask = torch.ones_like(img_attention_mask, dtype=torch.bool)  # NOTE: whole sequence is uncausally masked

            meta_attention_mask = torch.ones((B, N_METAWEIGHTS), device=device, dtype=torch.long)
            meta_uncausal_mask = torch.ones_like(meta_attention_mask, dtype=torch.bool)

            query_attention_mask = torch.ones((B, Sq), device=device, dtype=torch.long)
            query_uncausal_mask = torch.ones_like(query_attention_mask, dtype=torch.bool)


            # span ixs of metaweights to parse out (inclusive span). currently
            # lor_ixs is only defined for LOR_LAYER, other spans are None.
            lor_ixs = torch.zeros(B, 2, dtype=torch.long, device=device)
            with torch.no_grad():
                lor_ixs[:, 0] = Ss
                lor_ixs[:, 1] = Ss + N_METAWEIGHTS - 1

            lor_ixs_per_layer = (
                [None] * LOR_LAYER +
                [lor_ixs]  +
                [None] * (num_layers - LOR_LAYER - 1)
            )

            #####
            # Run supports and metaweights to populate lor_cache


            # iterate over supports and do SGD on LoRs
            with torch.enable_grad():
                opt_state = None  # will set first time this is needed

                # First pass to collect LORS
                empty_lor_cache = empty_lors(model.config.num_hidden_layers)  # init lors for whole batch
                out = model(inputs_embeds=torch.cat([support_imgs, metaweights], dim=1),  # note: input_embeds, not input_ids
                            attention_mask=torch.cat([img_attention_mask, meta_attention_mask], dim=1),
                            return_dict=True,
                            output_hidden_states=True,
                            uncausal_mask=torch.cat([img_uncausal_mask, meta_uncausal_mask], dim=1),
                            **empty_lor_cache,
                            )
                lor_cache = update_lors(lor_models, empty_lor_cache, lor_ixs_per_layer, out.hidden_states, num_layers)

                # N steps of metalearning
                for i in range(N_LOOPS):

                    # make sure lors require grads
                    for k in lor_cache.keys():
                        for lix in range(len(lor_cache[k])):
                            if lor_cache[k][lix] is not None:
                                lor_cache[k][lix] = [lor_cache[k][lix][0].requires_grad_(), lor_cache[k][lix][1].requires_grad_()]

                    lorm = partially_apply_models(lor_models, lor_cache)

                    # Second pass to train LORS
                    out = model(inputs_embeds=support_imgs,  # note: input_embeds, not input_ids
                                attention_mask=img_attention_mask,
                                return_dict=True,
                                output_hidden_states=True,
                                uncausal_mask=img_uncausal_mask,
                                **lorm,
                                )

                    # calculate TRANSDUCTIVE loss, ie not autoregressive, ie don't offset logits/targets (and don't causal mask)
                    logits = out.logits.contiguous().view(-1, vocab_size)  # note: not offset. just parses out img responses (not metaweights)
                    target = support_labels.view(-1)
                    loss = F.cross_entropy(logits, target)

                    key_ixs = []  # [(lor_key, layer_ix)]
                    params = []
                    # flatten params to be used in optim
                    for k in lor_cache.keys():
                        for lix in range(len(lor_cache[k])):
                            if lor_cache[k][lix] is not None:
                                for tuple_ix in range(2):
                                    key_ixs.append((k, lix, tuple_ix))
                                    params.append(lor_cache[k][lix][tuple_ix].requires_grad_())

                    grads = torch.autograd.grad(loss, params, create_graph=True)

                    MAX_GRAD_NORM = 1.0
                    grads = [g.renorm(2, dim=-1, maxnorm=MAX_GRAD_NORM) for g in grads]

                    if i == N_LOOPS - 1:
                        print(f'iloss: {loss.item():>.6f}')

                    if opt_state is None:
                        match INNER_OPT:
                            case 'sgd_momentum':
                                opt_state = sgd_momentum_init(params)
                            case 'rmsprop':
                                opt_state = rmsprop_init(params)
                            case 'adam':
                                opt_state = adam_init(params)
                            case 'adamax':
                                opt_state = adamax_init(params)

                    match INNER_OPT:
                        case 'sgd':
                            new_params = sgd(params, grads, lr=inner_lr)
                        case 'sgd_momentum':
                            new_params, opt_state = sgd_momentum(params, grads, opt_state, lr=inner_lr)
                        case 'rmsprop':
                            new_params, opt_state = rmsprop(params, grads, opt_state, lr=inner_lr)
                        case 'adam':
                            new_params, *opt_state = adam(params, grads, *opt_state, lr=inner_lr)
                        case 'adamax':
                            new_params, *opt_state = adamax(params, grads, *opt_state, lr=inner_lr)

                    lor_cache = empty_lors(model.config.num_hidden_layers)
                    for (k, lix, tuple_ix), p in zip(key_ixs, new_params):
                        if lor_cache[k][lix] is None:
                            lor_cache[k][lix] = [None, None]  # these will both be filled in
                        lor_cache[k][lix][tuple_ix] = p





            # #####
            # # QUERY with lor_cache, but no attention to original inputs (answers must live in weights ie lor_cache)

            # lorm = partially_apply_models(lor_models, lor_cache)

            # SQ = query_imgs.shape[1]
            # q_attention_mask = torch.ones(B, SQ, device=device, dtype=torch.long)
            # query_out = model(
            #     inputs_embeds=query_imgs,
            #     attention_mask=q_attention_mask,
            #     **lorm,
            # )

            # logits = query_out.logits[:, :-1].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
            # target = query_labels[:, 1:].contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
            # qloss = F.cross_entropy(logits, target)

            # # final metalearning loss + query loss
            # loss = loss + qloss


            # #####
            # # SHAM QUERY, re-run support for final loss

            # lorm = partially_apply_models(lor_models, lor_cache)

            # SS = support_imgs.shape[1]
            # s_attention_mask = torch.ones(B, SS, device=device, dtype=torch.long)

            # query_out = model(
            #     inputs_embeds=support_imgs,
            #     attention_mask=s_attention_mask,

            #     # inputs_embeds=inputs_embeds,
            #     # attention_mask=img_attention_mask,
            #     # uncausal_mask=uncausal_mask,

            #     **lorm,
            # )

            # logits = query_out.logits[:, :S].contiguous().view(-1, vocab_size)  # [B, S-1, D] -> [B * (S-1), D]
            # target = support_labels.contiguous().view(-1)  # [B, S-1] -> [B * (S-1)]
            # qloss = F.cross_entropy(logits, target)

            # # final metalearning loss + query loss
            # # loss = loss + qloss
            # loss = qloss


            if torch.isnan(loss):
                print('NaN encountered:')
                # print(f'  all loss was masked out?: {sum([x.to(dtype=torch.long).sum() for x in loss_masks]) == 0}')
                # print(f'  nan in out_logits?: {torch.cat(out_logits, dim=1).isnan().sum() > 0}')
                # print('  ragged batch sizes? (double check by hand if necessary)')
                breakpoint()

            if train:
                optimizer.zero_grad()
                loss.backward()

                MAX_GRAD_NORM = 1.0
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(lor_models.parameters(), max_norm=MAX_GRAD_NORM)
                nn.utils.clip_grad_norm_(img_proj.parameters(), max_norm=MAX_GRAD_NORM)

                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        if grad_norm > 100:
                            print(f"Large gradient in {name}: {grad_norm}")

                optimizer.step()

            total_loss += loss.item() * B
            total_samples += B

    avg_loss = total_loss / total_samples

    # Log weight histogram
    if LOG and train:
        try:
            # for name, param in itertools.chain(model.named_parameters(), lor_models.named_parameters()):
            for name, param in lor_models.named_parameters():
                if param.requires_grad:
                    writer.add_histogram(f'weights/{name}', param.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
                    if param.grad is not None:
                        writer.add_histogram(f'grads/{name}', param.grad.data.detach().to(dtype=torch.float32).cpu().numpy(), global_epoch)
        except Exception as e:
            warnings.warn(f'Failed to write to tensorboard: {e}')

    if debug:
        warnings.warn('only debugging the last batch, values are not accumulated across batches (loss *is* averaged though)')
        return avg_loss, out, challenge_out, lor_cache
    else:
        return avg_loss




##################################################
# Go

global_epoch = 0

# data
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]
img_size = 28
n_way = 2  # N-way classification
k_shot = 1  # k-shot learning
q_query = 1  # query examples per class
num_tasks = 100  # number of tasks per epoch

# training
num_epochs = 100
batch_size = 32
lr = 1e-4
wd = 1e-2

# INNER_OPT = 'sgd'
INNER_OPT = 'sgd_momentum'
# INNER_OPT = 'rmsprop'
# INNER_OPT = 'adam'
# INNER_OPT = 'adamax'

match INNER_OPT:
    case 'sgd':
        INNER_LR = 1.0  # SGD-momentum needs much higher lr
        N_LOOPS = 20
    case 'sgd_momentum':
        INNER_LR = 1.0  # SGD-momentum needs much higher lr
        N_LOOPS = 20
    case 'rmsprop':
        INNER_LR = 1e-1
        N_LOOPS = 20
    case 'adam':
        INNER_LR = 1e-1
        N_LOOPS = 20
    case 'adamax':
        INNER_LR = 1e-2
        N_LOOPS = 20


##########
# LoR Models

dim = model.model.embed_tokens.weight.shape[1]
k_dim = model.model.layers[0].self_attn.k_proj.weight.shape[0]
v_dim = model.model.layers[0].self_attn.v_proj.weight.shape[0]
ff_dim = model.config.intermediate_size

# Note: there must be at least a None per each QKVOGUD block per layer
lor_models = nn.ModuleDict(
    {

        #####
        # Projection
        'lor_proj': nn.ModuleList([None] * num_layers),

        #####
        # Norms

        # low rank attention params
        "lor_qs": nn.ModuleList([None] * num_layers),
        "lor_ks": nn.ModuleList([None] * num_layers),
        "lor_vs": nn.ModuleList([None] * num_layers),
        "lor_os": nn.ModuleList([None] * num_layers),

        # low rank mlp params
        "lor_gs": nn.ModuleList([None] * num_layers),
        "lor_us": nn.ModuleList([None] * num_layers),
        "lor_ds": nn.ModuleList([None] * num_layers),
    }
)

lor_models['lor_proj'][LOR_LAYER] = LORProject()

if WHICH_LOR == 1:
    lor_models['lor_qs'][LOR_LAYER] = LORNorm(dim, dim, name='lor_qs')
    lor_models['lor_ks'][LOR_LAYER] = LORNorm(dim, k_dim, name='lor_ks')
    lor_models['lor_vs'][LOR_LAYER] = LORNorm(dim, v_dim, name='lor_vs')
    lor_models['lor_os'][LOR_LAYER] = LORNorm(dim, dim, name='lor_os')

    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim, name='lor_gs')
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim, name='lor_us')
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim, name='lor_ds')
elif WHICH_LOR == 2:
    lor_models['lor_gs'][LOR_LAYER] = LORNorm(dim, ff_dim, name='lor_gs')
    lor_models['lor_us'][LOR_LAYER] = LORNorm(dim, ff_dim, name='lor_us')
    lor_models['lor_ds'][LOR_LAYER] = LORNorm(dim, dim, name='lor_ds')
lor_models = lor_models.to(DEVICE, dtype=model.dtype)


print_model_info(lor_models)
add_hooks(lor_models)


##########
# Image Projection

img_proj = nn.Sequential(
    nn.Linear(img_size ** 2, dim),
    nn.LayerNorm(dim)
)
with torch.no_grad():
    assert isinstance(img_proj[1], nn.LayerNorm)
    img_proj[1].weight[:] = torch.ones_like(img_proj[1].weight) * 1e-1

img_proj.to(DEVICE)

##########
# Dataset

train_dl, test_dl = omniglot_n_way_k_shot(
    train_alphabets,
    test_alphabets,
    n_way,
    k_shot,
    q_query,
    num_tasks,
    img_size,
    batch_size,
)


#####
# Check Data

if False:
    # START_BLOCK_1

    # Explore Data
    for batch in train_dl:
        supports, queries = batch
        support_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in supports], dim=1)  # N*k tensors [B, IMG, IMG]
        support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG]
        query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        break



    # We'll visualize the first batch only
    batch_idx = 0

    # Calculate image size (assuming square images)
    img_size = int(np.sqrt(support_imgs.shape[-1]))

    # Create a figure with a grid: k_shot + 1 rows (support + query) x n_way columns
    fig, axes = plt.subplots(k_shot + 1, n_way, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.5)

    # Plot support images
    for i in range(k_shot):
        for j in range(n_way):
            idx = i * n_way + j
            img = support_imgs[batch_idx, idx].cpu().numpy().reshape(img_size, img_size)
            label = support_labels[batch_idx, idx].cpu().numpy()

            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Class {label}')

    # Plot query images
    for j in range(n_way):
        img = query_imgs[batch_idx, j].cpu().numpy().reshape(img_size, img_size)
        label = query_labels[batch_idx, j].cpu().numpy()

        axes[-1, j].imshow(img, cmap='gray')
        axes[-1, j].axis('off')
        axes[-1, j].set_title(f'Query\nClass {label}')

    # Add row labels
    for i in range(k_shot):
        axes[i, 0].set_ylabel(f'Support\nSet {i+1}', rotation=0, labelpad=40)
    axes[-1, 0].set_ylabel('Query\nSet', rotation=0, labelpad=40)

    plt.suptitle('Omniglot Few-Shot Learning Task Visualization\n'
                 f'{n_way}-way {k_shot}-shot with 1 query example per class',
                 y=1.02)
    plt.tight_layout()

    plt.show()

    # END_BLOCK_1




##########
# Optimizer

parameters = [

{
    'params': model.parameters(),
    'lr': 1e-5,
    'wd': 0.0
},

{
    'params': img_proj.parameters(),
    'lr': lr,
    'wd': wd
},

{
    'params': lor_models.parameters(),
    'lr': lr,
    'wd': wd
}
]

optimizer = optim.AdamW(parameters)

####################

train_losses = []
test_losses = []
best_loss = float('inf')


for epoch in range(num_epochs):
    global_epoch += 1

    # with torch.autograd.detect_anomaly():
    model.train()
    train_loss = run_epoch(model, img_proj, lor_models, INNER_LR, train_dl, optimizer, DEVICE, train=True)

    train_losses.append(train_loss)
    if LOG: writer.add_scalars('loss', {'train': train_loss}, global_epoch)

    if epoch % 5 == 0:
        model.eval()
        with torch.no_grad():
            test_loss = run_epoch(model, img_proj, lor_models, INNER_LR, test_dl, optimizer, DEVICE, train=False)
        test_losses.append(test_loss)

        if LOG: writer.add_scalars('loss', {'test': test_loss}, global_epoch)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    else:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")



if num_epochs > 0:
    l = len(test_losses)
    epochs = list(range(l))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses[:l], label='Training Loss', color='blue')
    plt.plot(epochs, test_losses, label='Test Loss', color='red')

    # Customize the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()



##################################################
# Analyze trained params

for k in lor_models.keys():
    for lix in range(len(lor_models[k])):
        if lor_models[k][lix] is None:
            continue
        for n, p in lor_models[k][lix].named_parameters():
            print()
            print(f'{k} | {n}')
            xs = [f'{x:>.3f}' for x in p.flatten()[:10].tolist()]
            print(f'{p.min().item():>.3f}, {p.max().item():>.3f}, {p.mean().item():>.3f}, {p.std().item():>.3f}, {xs}')


'''

lor_proj | token_mixing_mlp.0.weight
1.000, 1.000, 1.000, 0.000, ['1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000', '1.000']

lor_proj | token_mixing_mlp.1.weight
-0.267, 0.266, -0.002, 0.153, ['0.200', '-0.127', '-0.068', '0.026', '0.167', '0.043', '-0.109', '0.084', '-0.182', '-0.058']

lor_proj | token_mixing_mlp.4.weight
-0.088, 0.088, 0.001, 0.051, ['0.079', '-0.015', '-0.046', '-0.066', '-0.068', '-0.031', '-0.022', '-0.029', '-0.008', '-0.036']



lor_proj | channel_mixing_mlp.0.weight
0.997, 1.002, 1.000, 0.001, ['1.000', '1.000', '0.999', '0.999', '0.999', '0.998', '1.000', '0.999', '0.998', '0.999']

lor_proj | channel_mixing_mlp.1.weight
-0.036, 0.036, -0.000, 0.019, ['0.012', '0.012', '0.005', '0.004', '-0.022', '0.026', '0.032', '0.016', '-0.028', '-0.023']

lor_proj | channel_mixing_mlp.4.weight
-0.090, 0.090, 0.000, 0.051, ['-0.045', '-0.015', '-0.001', '-0.007', '-0.005', '0.068', '-0.049', '0.084', '0.028', '-0.019']



lor_proj | final_projections.lor_qs_l.weight
-0.033, 0.033, -0.000, 0.019, ['0.024', '-0.005', '-0.022', '0.033', '0.018', '-0.019', '-0.017', '0.031', '-0.021', '0.012']

lor_proj | final_projections.lor_qs_r.weight
-0.033, 0.033, -0.000, 0.019, ['0.011', '0.026', '-0.011', '0.030', '0.002', '-0.001', '-0.027', '0.011', '0.032', '-0.017']

lor_proj | final_projections.lor_ks_l.weight
-0.033, 0.033, 0.000, 0.019, ['-0.029', '-0.008', '0.025', '0.010', '0.022', '0.019', '0.028', '-0.006', '0.011', '0.023']

lor_proj | final_projections.lor_ks_r.weight
-0.033, 0.033, -0.000, 0.019, ['-0.000', '0.011', '-0.019', '0.032', '0.030', '0.020', '0.003', '-0.031', '-0.012', '-0.031']

lor_proj | final_projections.lor_vs_l.weight
-0.035, 0.035, -0.000, 0.019, ['-0.015', '0.016', '0.014', '-0.021', '-0.015', '-0.012', '-0.028', '0.030', '-0.021', '0.009']

lor_proj | final_projections.lor_vs_r.weight
-0.035, 0.035, -0.000, 0.019, ['0.025', '0.004', '-0.027', '-0.011', '0.011', '-0.022', '0.017', '-0.013', '0.001', '-0.029']

lor_proj | final_projections.lor_os_l.weight
-0.036, 0.037, -0.000, 0.019, ['-0.027', '0.023', '-0.030', '-0.030', '-0.031', '-0.016', '0.003', '0.014', '0.005', '0.010']

lor_proj | final_projections.lor_os_r.weight
-0.036, 0.036, -0.000, 0.019, ['-0.009', '-0.010', '0.001', '0.013', '-0.019', '0.030', '0.013', '0.005', '-0.032', '0.027']

lor_proj | final_projections.lor_gs_l.weight
-0.035, 0.035, -0.000, 0.019, ['-0.017', '0.004', '-0.004', '-0.026', '-0.021', '0.028', '0.032', '0.007', '0.031', '0.010']

lor_proj | final_projections.lor_gs_r.weight
-0.035, 0.035, 0.000, 0.019, ['-0.026', '-0.003', '-0.013', '0.017', '-0.001', '0.020', '-0.028', '0.029', '0.007', '0.006']

lor_proj | final_projections.lor_us_l.weight
-0.033, 0.033, -0.000, 0.019, ['-0.015', '0.001', '0.021', '0.033', '0.001', '0.014', '0.010', '-0.002', '0.033', '0.031']

lor_proj | final_projections.lor_us_r.weight
-0.035, 0.035, 0.000, 0.019, ['-0.025', '0.021', '0.025', '0.005', '0.006', '-0.012', '0.017', '-0.019', '-0.021', '-0.032']

lor_proj | final_projections.lor_ds_l.weight
-0.037, 0.037, -0.000, 0.019, ['0.011', '0.013', '0.019', '-0.018', '0.009', '-0.019', '-0.034', '0.017', '0.023', '0.015']

lor_proj | final_projections.lor_ds_r.weight
-0.033, 0.033, -0.000, 0.019, ['-0.021', '-0.021', '0.019', '-0.027', '-0.017', '-0.033', '-0.010', '0.010', '0.009', '-0.021']



lor_qs | norm.weight
0.009, 0.011, 0.010, 0.000, ['0.010', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010']

lor_ks | norm.weight
0.009, 0.011, 0.010, 0.000, ['0.010', '0.011', '0.011', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010', '0.010']

lor_vs | norm.weight
0.008, 0.012, 0.010, 0.001, ['0.010', '0.009', '0.010', '0.010', '0.010', '0.011', '0.010', '0.010', '0.009', '0.011']

lor_os | norm.weight
0.009, 0.015, 0.012, 0.001, ['0.012', '0.010', '0.012', '0.013', '0.010', '0.013', '0.010', '0.014', '0.012', '0.014']

lor_gs | norm.weight
0.008, 0.013, 0.010, 0.001, ['0.011', '0.011', '0.009', '0.011', '0.011', '0.009', '0.011', '0.010', '0.011', '0.010']

lor_us | norm.weight
0.008, 0.013, 0.010, 0.001, ['0.011', '0.011', '0.009', '0.011', '0.011', '0.009', '0.011', '0.011', '0.011', '0.010']

lor_ds | norm.weight
0.009, 0.016, 0.012, 0.001, ['0.012', '0.012', '0.011', '0.011', '0.011', '0.012', '0.012', '0.014', '0.010', '0.014']



'''
