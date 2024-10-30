'''

A simple homoiconic MLP for solving the N-way k-shot task


>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1)(torch.randn((3, 5, 7, 11))).shape
torch.Size([3, 13, 5, 9])

>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1, padding=1).weight.shape
torch.Size([3, 13, 7, 11])

>>> nn.Conv2d(in_channels=5, out_channels=13, kernel_size=3, stride=1, padding=1).weight.shape
torch.Size([13, 5, 3, 3])

'''

import os
import math
# import sys
import traceback
import importlib
import random
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import warnings
from functools import partial
import itertools
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer

import numpy as np
import matplotlib.pyplot as plt

from t13_metalearning_hypernet_data import omniglot_n_way_k_shot, ALPHABET_DICT
from neurallambda.lab.common import print_model_info
from torch.autograd import Function

from einops import rearrange, repeat

SEED = 152
torch.manual_seed(SEED)
random.seed(SEED)

DEVICE = 'cuda'


##################################################
# Params

img_dim = 28
D = 64
label_dim = D


# data
B = 32
train_alphabets = ["Latin", "Greek"]
test_alphabets = ["Mongolian"]
img_size = 28
n_way = 2  # N-way classification
k_shot = 1  # k-shot learning
q_query = 1  # query examples per class
num_tasks = B * 50  # number of tasks per epoch



##################################################
# Batchified functions

def batch_linear(input, weight, bias=None):
    ''' uses batch-wise different weights and biases '''
    output = torch.bmm(input.unsqueeze(1), weight.transpose(1, 2)).squeeze(1)
    if bias is not None:
        output += bias
    return output


##################################################
# Functional Optimizers
#
#   Differentiable variants of: SGD, SGD-Momentum, RMSProp, Adam


##########
# SGD

def sgd_init(params):
    # just used to maintain call-signature parity with other optimizers
    return []

def sgd(params, grads, opt_state: List[Any], lr=0.01):
    # opt_state is ignored. included to maintain signature parity with other optimizers
    return (
        [p - g * lr for p, g in zip(params, grads)],
        []  # empty opt_state
    )


##########
# SGD Momentum

def sgd_momentum_init(params):
    return [torch.zeros_like(p) for p in params]

def sgd_momentum(params, grads, opt_state: List[Any], lr=0.01, momentum=0.9):
    velocity = opt_state[0]
    updated_velocity = [momentum * v + lr * g for v, g in zip(velocity, grads)]
    updated_params = [p - v for p, v in zip(params, updated_velocity)]
    return updated_params, updated_velocity


##########
# RMSProp

def rmsprop_init(params):
    return [torch.zeros_like(p) for p in params]  # square averages

def rmsprop(params, grads, opt_state: List[Any], lr=0.01, alpha=0.99, eps=1e-8):
    square_avg = opt_state[0]
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

def adam(params, grads, opt_state, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
    m, v, t = opt_state
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

def adamax(params, grads, opt_state, lr=0.002, betas=(0.9, 0.999), eps=1e-8):
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
    m, u, t = opt_state
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
# Data

global_epoch = 0


# training
num_epochs = 1000
lr = 1e-4
wd = 1e-2
alpha = 1e-1  # scale of weight_loss loss


##########
# Dataset

try:
    already_loaded
except:
    train_dl, test_dl = omniglot_n_way_k_shot(
        train_alphabets,
        test_alphabets,
        n_way,
        k_shot,
        q_query,
        num_tasks,
        img_size,
        B,
        seed_train=1,
        seed_test=2,
    )
    already_loaded = True

   #  # TODO: remove hack. The omniglot dataset has separate train/test splits,
   #  # but if I want random label pairings, but keep the same alphabet across
   #  # train/test, I need to do this dumb hack rn.
   #  train_dl, _ = omniglot_n_way_k_shot(
   #      train_alphabets,
   #      ["Mongolian"],  # not used
   #      n_way,
   #      k_shot,
   #      q_query,
   #      num_tasks,
   #      img_size,
   #      B,
   #      seed_train=1,
   #      seed_test=2,
   #  )

   #  test_dl, _ = omniglot_n_way_k_shot(
   #      test_alphabets,
   #      ["Mongolian"],  # not used
   #      n_way,
   #      k_shot,
   #      q_query,
   #      num_tasks,
   #      img_size,
   #      B,
   #      seed_train=3,
   #      seed_test=4,
   # )

    already_loaded = True


if False:
    # Explore Data
    for batch in train_dl:
        supports, queries = batch
        # support_imgs.shape   = [32, 2, 784]
        # support_labels.shape = [32, 2]
        # query_imgs.shape     = [32, 2, 784]
        # query_labels.shape   = [32, 2]
        support_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in supports], dim=1)  # N*k tensors [B, IMG, IMG]
        support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG]
        query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        break


if False:
    # START_BLOCK_4

    # Explore Data
    # for batch in train_dl:
    for batch in test_dl:
        supports, queries = batch
        support_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in supports], dim=1)  # N*k tensors [B, IMG, IMG]
        support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = torch.stack([x[0].to(DEVICE).flatten(start_dim=1, end_dim=2) for x in queries], dim=1)  # N*k tensors [B, IMG, IMG]
        query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # N*k tensors, shape=[B] -> [B, N*k]
        break

    # We'll visualize the first batch only
    batch_idx = 6

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

    # END_BLOCK_4



##################################################

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        output = torch.mm(input, weight.t()) + bias
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        grad_input = torch.mm(grad_output, weight)
        grad_weight = torch.mm(grad_output.t(), input)
        grad_bias = grad_output.sum(dim=0)

        return grad_input, grad_weight, grad_bias

class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return LinearFunction.apply(x, self.weight, self.bias)



##################################################


# START_BLOCK_1

class RandomScale(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        if not self.training:
            return x
        s = torch.rand_like(x) * self.scale
        return x * s

class Model(nn.Module):
    def __init__(self, dim, img_dim):
        super().__init__()
        self.dim = dim

        self.img_in = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((16, 16)), nn.BatchNorm2d(8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((8, 8)), nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, stride=1, padding=1), nn.GELU(), nn.AdaptiveAvgPool2d((8, 8)), nn.BatchNorm2d(1),

            nn.Flatten(1, -1),
            nn.Dropout(0.2),
            nn.Linear(8 ** 2, dim, bias=False),
        )
        self.emb = nn.Parameter(torch.randn(n_way, label_dim))
        torch.nn.init.orthogonal_(self.emb)

        self.w1 = nn.Parameter(torch.randn(dim, dim))
        torch.nn.init.orthogonal_(self.w1)

        self.w2 = nn.Parameter(torch.randn(dim, dim))
        torch.nn.init.orthogonal_(self.w2)

    def net(self, x):
        x = torch.einsum('ed, ...d -> ...e', self.w1, x)
        x = F.gelu(x)
        x = torch.einsum('ed, ...d -> ...e', self.w2, x)
        return x

    def forward(self,
                query_w,
                sxs, sys,  # support xs/ys
                qxs, qys,  # query xs/ys
                inner_lr):
        B = query_w.shape[0]
        S = sxs.shape[1]

        # embed inputs
        sxs    = self.img_in(sxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * k_shot, -1)  # [B, S, D]
        sys_emb = self.emb[sys]

        # embed outputs
        qxs = self.img_in(qxs.view(-1, 1, img_dim, img_dim)).view(B, n_way * q_query, -1)  # [B, Q, D]
        qys_emb = self.emb[qys]

        # with torch.enable_grad():
        #     opt_state = None
        #     w = (self.w1.unsqueeze(0).repeat(B, 1, 1) + 0).requires_grad_()
        if False:
            # for _ in range(N_LOOPS):
            #     # pass images through w
            #     pxs = torch.einsum('bed, bnd -> bne', w, sxs)
            #     # build outer pdt
            #     op = torch.einsum('bsd, bse -> bde', pxs, sys_emb)  # outer pdt of images and labels

            #     # pass images through outer pdt
            #     pred_labels = torch.einsum('bde, bnd -> bne', op, pxs)

            #     # predict label
            #     pred_labels = torch.einsum('ld, bnd -> bnl', self.emb, pred_labels)
            #     loss = F.cross_entropy(
            #         pred_labels.view(-1, n_way),
            #         sys.view(-1)
            #     )
            #     grads = torch.autograd.grad(loss, w, create_graph=True)

            # opt_state = opt_state_fn(w)
            # new_params, opt_state = opt_fn(w, grads, opt_state, lr=inner_lr)

            #     w = new_params[0]

            pass


        w = 0
        for six in range(S):
            # w = w + self.net(torch.cat([sxs[:, six], sys_emb[:, six]], dim=1))
            # w = w + self.net(sxs[:, six] * sys_emb[:, six])
            w = w + torch.einsum('bd, be -> bde', self.net(sxs[:, six]), self.net(sys_emb[:, six]))

        w = w.reshape(B, D, D)

        y = torch.einsum('bde, bqd -> bqe', w, qxs)
        pred_labels = torch.einsum('ld, bqd -> bql', self.emb, y)

        # l1 = torch.einsum('de, be -> bd', self.w1.weight, query_w)
        # r1 = torch.einsum('de, bd -> be', self.w1.weight, query_w)

        # l1 = self.w1(sxs).mean(dim=1)
        # r1 = self.w1(sys_emb).mean(dim=1)

        l1 = (sxs).mean(dim=1)
        r1 = (sys_emb).mean(dim=1)

        assert pred_labels.shape[0] == B
        assert pred_labels.shape[1] == qys.shape[1]
        assert pred_labels.shape[2] == n_way

        w_recon = torch.einsum('bl, br -> blr', l1, r1)
        w_loss = F.mse_loss(w_recon, model.w1.unsqueeze(0).repeat(B, 1, 1))

        task_loss = F.cross_entropy(
            pred_labels.view(-1, n_way),
            qys.view(-1)
        )

        loss = task_loss + alpha * w_loss


        return loss, task_loss, w_loss



##################################################

num_epochs = 1000

# INNER_OPT = 'sgd'
INNER_OPT = 'sgd_momentum'
# INNER_OPT = 'rmsprop'
# INNER_OPT = 'adam'
# INNER_OPT = 'adamax'





match INNER_OPT:
    case 'sgd':
        INNER_LR = 1.0  # SGD-momentum needs much higher lr
        N_LOOPS = 20
        opt_state_fn = sgd_init
        opt_fn = sgd
    case 'sgd_momentum':
        INNER_LR = 0.1  # SGD-momentum needs much higher lr
        N_LOOPS = 5
        opt_state_fn = sgd_momentum_init
        opt_fn = sgd_momentum
    case 'rmsprop':
        INNER_LR = 1e-1
        N_LOOPS = 20
        opt_state_fn = rmsprop_init
        opt_fn = rmsprop
    case 'adam':
        INNER_LR = 1e-1
        N_LOOPS = 20
        opt_state_fn = adam_init
        opt_fn = adam
    case 'adamax':
        INNER_LR = 1e-2
        N_LOOPS = 20
        opt_state_fn = adamax_init
        opt_fn = adamax


model = Model(D, img_dim).to(DEVICE)

print_model_info(model)

# parameters = [
#     # model.w_in[0].weight,
#     # model.img_in[1].weight,
#     model.emb,
#     model.w1,
#     # model.w_out.weight,
# ]

parameters = model.parameters()

optimizer = optim.AdamW(parameters, lr=lr, weight_decay=wd)

def run_epoch(model, dataloader, inner_lr, optimizer, device, train=True):
    model.train() if train else model.eval()
    total_loss = 0
    total_task_loss = 0
    total_weight_loss = 0
    total_samples = 0

    with torch.set_grad_enabled(train):
        for batch in dataloader:
            supports, queries = batch
            support_imgs = torch.stack([x[0].to(DEVICE) for x in supports], dim=1)  # [B, NK, IMG**2]
            support_labels = torch.stack([x[1].to(DEVICE) for x in supports], dim=1)  # [B, NK]
            query_imgs = torch.stack([x[0].to(DEVICE) for x in queries], dim=1)  # [B, NK, IMG**2]
            query_labels = torch.stack([x[1].to(DEVICE) for x in queries], dim=1)  # [B, NQ]

            B = support_imgs.shape[0]
            query_w = torch.randn((B, D)).to(DEVICE)

            loss, task_loss, w_loss = model(query_w, support_imgs, support_labels, query_imgs, query_labels, inner_lr)

            if torch.isnan(loss):
                print('NaN encountered:')
                breakpoint()

            if train:
                MAX_GRAD_NORM = 1.0

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()


            total_loss += loss.item() * B
            total_task_loss += task_loss.item() * B
            total_weight_loss += w_loss.item() * B
            total_samples += B

    return (
        total_loss / total_samples,
        total_task_loss / total_samples,
        total_weight_loss / total_samples
    )


train_losses = []
test_losses = []

for epoch in range(num_epochs):
    global_epoch += 1

    # with torch.autograd.detect_anomaly():
    model.train()
    train_loss, train_task_loss, train_weight_loss = run_epoch(model, train_dl, INNER_LR, optimizer, DEVICE, train=True)

    train_losses.append((global_epoch, train_loss))

    if epoch % 40 == 0:
        model.eval()
        with torch.no_grad():
            test_loss, test_task_loss, test_weight_loss = run_epoch(model, test_dl, INNER_LR, optimizer, DEVICE, train=False)
        test_losses.append((global_epoch, test_loss))
        print(f"Epoch {global_epoch}/{num_epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
        print(f'    TRAIN: task_loss: {train_task_loss:>.3f}, w_loss: {train_weight_loss:>.3f}')
        print(f'    TEST:  task_loss: {test_task_loss:>.3f}, w_loss: {test_weight_loss:>.3f}')

    elif epoch % 10 == 0:
        print(f"Epoch {global_epoch}/{num_epochs}, Train Loss: {train_loss:.4f}")


# END_BLOCK_1



weight_1 = model.w1.detach().cpu().numpy()

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
axs[0].imshow(weight_1)
fig.tight_layout()
plt.show()





#####
# EXPERIMENT: Instead of using a net, or training, just outer pdt imgs and labels, and check cossim of query with target label. Works great!
#
# NOTE: this has to learn a single matrix *across the entire batch* and it still works

if False:

    label_dim = 28 ** 2
    emb = nn.Embedding(n_way, label_dim).to(DEVICE)
    for batch in train_dl:
        supports, queries = batch
        support_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in supports]  # N*k tensors [B, IMG, IMG]
        support_labels = [x[1].to(DEVICE) for x in supports]  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in queries]  # N*k tensors [B, IMG, IMG]
        query_labels = [x[1].to(DEVICE) for x in queries]  # N*k tensors, shape=[B] -> [B, N*k]
        w = torch.zeros(label_dim, img_size**2).to(DEVICE)
        for im, lab in zip(support_imgs, support_labels):
            w += torch.einsum('bd, bl -> ld', im, emb(lab))
        cs = 0
        correct = 0
        for im, lab in zip(query_imgs, query_labels):
            out = torch.einsum('ld, bd -> bl', w, im)
            cs += torch.cosine_similarity(out, emb(lab), dim=1)
            pred = out.argmax(dim=1)
            correct += (pred == lab).sum()
        cs /= len(query_imgs)
        print(cs.mean().item())
        print(correct)

        brk

if False:
    # START_BLOCK_3
    def norm(vectors):
        means = vectors.mean(dim=1, keepdim=True)
        centered = vectors - means
        # stds = torch.sqrt(torch.var(centered, dim=1, keepdim=True) + 1e-8)
        stds = centered.abs().max()
        return centered / stds


    def blur_batch(vectors, kernel):
        # Reshape kernel for batch convolution: [1, 1, kernel_size]
        kernel = kernel.view(1, 1, -1)

        # Add channel dim to input: [batch, 1, vector_length]
        x = vectors.unsqueeze(1)

        # Apply convolution with padding to maintain size
        blurred = F.conv1d(x, kernel, padding=kernel.size(-1) // 2)

        # Remove channel dim
        return blurred.squeeze(1)


    label_dim = 64
    emb = nn.Embedding(n_way, label_dim)
    nn.init.orthogonal_(emb.weight)
    emb = emb.to(DEVICE)
    with torch.no_grad():
        emb.weight[:] = norm(emb.weight)

    for batch in train_dl:
        supports, queries = batch

        # Process support images and labels
        support_imgs = [norm(x[0].to(DEVICE).flatten(start_dim=1)) for x in supports]
        support_labels = [x[1].to(DEVICE) for x in supports]

        # Process query images and labels
        query_imgs = [norm(x[0].to(DEVICE).flatten(start_dim=1)) for x in queries]
        query_labels = [x[1].to(DEVICE) for x in queries]

        B = support_imgs[0].shape[0]

        # Initialize and compute weight matrix
        w = torch.zeros(B, label_dim, img_size**2).to(DEVICE)
        for _ in range(100):
            for im, lab in zip(support_imgs, support_labels):
                lab_emb = emb(lab)
                kernel = torch.rand(3).to(DEVICE)
                w = w + torch.einsum(
                    'bl, bi -> bli',
                    lab_emb,
                    im + blur_batch(im, kernel)
                )

        # Evaluate queries
        correct = 0
        total = 0
        all_similarities = []

        for im, lab in zip(query_imgs, query_labels):
            # Project query through weight matrix
            projected = torch.einsum('bli, bi -> bl', w, im)

            # Calculate similarities with all possible classes
            similarities = torch.einsum('bl, kl -> bk', projected, emb.weight)

            # Get predictions
            pred = similarities.argmax(dim=1)
            correct += (pred == lab).sum().item()
            total += lab.size(0)

            all_similarities.append(similarities)

        # Calculate accuracy
        accuracy = correct / total

        # Calculate loss
        all_similarities = torch.cat(all_similarities, dim=0)
        all_labels = torch.cat(query_labels, dim=0)
        loss = F.cross_entropy(all_similarities, all_labels)

        print(f"Accuracy: {accuracy:.4f}, Loss: {loss.item():.4f}")
    # END_BLOCK_3




#####
# Experiment with CONVOLUTION

# START_BLOCK_2

if False:
    emb = nn.Embedding(n_way, label_dim).to(DEVICE)
    for batch in train_dl:
        supports, queries = batch
        # Normalize images first
        support_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in supports]  # N*k tensors [B, IMG*IMG]
        support_labels = [x[1].to(DEVICE) for x in supports]  # N*k tensors, shape=[B] -> [B, N*k]
        query_imgs = [F.normalize(x[0].to(DEVICE).flatten(start_dim=1, end_dim=2), dim=1) for x in queries]  # N*k tensors [B, IMG*IMG]
        query_labels = [x[1].to(DEVICE) for x in queries]  # N*k tensors, shape=[B] -> [B, N*k]

        B = support_imgs[0].shape[0]
        Q = len(query_labels)

        # Stack all support images and labels
        support_imgs = torch.stack(support_imgs, dim=1)  # [B, N*k, D]
        support_labels = torch.stack(support_labels, dim=1)  # [B, N*k]
        query_imgs = torch.stack(query_imgs, dim=1)  # [B, Q, D]
        query_labels = torch.stack(query_labels, dim=1)  # [B, Q]

        # Get normalized embeddings for all labels
        support_emb = F.normalize(emb(support_labels), dim=-1)  # [B, N*k, D]

        # Compute all support convolutions at once using conjugate for correct correlation

        # fft_support_imgs = torch.fft.fft(support_imgs, dim=-1)  # [B, N*k, D]
        # fft_support_emb = torch.fft.fft(support_emb, dim=-1)  # [B, N*k, D]
        # support_bound = torch.fft.ifft(fft_support_imgs * fft_support_emb, dim=-1).real  # [B, N*k, D]

        # simple mul, instead of convolution
        support_bound = support_imgs * support_emb

        # Normalize the bound vectors
        support_bound = F.normalize(support_bound, dim=-1)

        # Pre-compute query convolutions with all possible class embeddings
        all_classes = torch.arange(n_way).to(DEVICE)
        class_emb = F.normalize(emb(all_classes), dim=-1)  # [N, D]

        # fft_query = torch.fft.fft(query_imgs, dim=-1).unsqueeze(-2)  # [B, Q, 1, D]
        # fft_class_emb = torch.fft.fft(class_emb, dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, D]
        # query_bound = torch.fft.ifft(fft_query * fft_class_emb, dim=-1).real  # [B, Q, N, D]

        # simple mul, instead of convolution
        query_bound = (
            query_imgs.unsqueeze(2) *
            class_emb.unsqueeze(0).unsqueeze(0)
        )

        # Normalize the query bound vectors
        query_bound = F.normalize(query_bound, dim=-1)

        # Compute logits for all queries and all possible classes
        logits = torch.zeros((B, Q, n_way)).to(DEVICE)

        # Debug distributions
        class_counts = torch.zeros(n_way)

        # Compute similarities with all support examples
        for class_idx in range(n_way):
            mask = (support_labels == class_idx)  # [B, N*k]
            masked_support = support_bound * mask.unsqueeze(-1)  # [B, N*k, D]
            class_prototype = masked_support.sum(dim=1) / (mask.sum(dim=1, keepdim=True) + 1e-8)  # [B, D]

            # Normalize the class prototype
            class_prototype = F.normalize(class_prototype, dim=-1)

            # Use the same class index to align with the class embeddings used in query_bound
            sim = torch.einsum(
                'bqnd, bd -> bqn',
                query_bound,  # [B, Q, N, D]
                class_prototype  # [B, D]
            )[:, :, class_idx]  # Take the similarity for the current class

            logits[:, :, class_idx] = sim

        loss = F.cross_entropy(logits.view(-1, n_way), query_labels.view(-1))

        # Compute accuracy
        predictions = logits.argmax(dim=1)  # [B, Q]
        correct = (predictions == query_labels).float()
        accuracy = correct.mean().item()

        # Print per-class prediction distribution
        print()
        for i in range(n_way):
            count = (predictions == i).float().mean().item()
            print(f'Class {i} predicted: {count:.2%}')

        print(f'Loss: {loss.item():.4f}, Accuracy: {accuracy:.2%}')
        break


# END_BLOCK_2
