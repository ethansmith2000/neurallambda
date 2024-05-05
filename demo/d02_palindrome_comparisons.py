'''

Compare architectures on this task using RayTune

'''

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # quiet the tokenizers warning


import ray
from ray import tune
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock

import neurallambda.stack as S
from neurallambda.lab.common import (
    build_tokenizer_dataloader,
    dataloader_info,
    print_grid,
    print_model_info,
    run_epoch,
)
from neurallambda.torch import GumbelSoftmax


##################################################
# PARAMS

DEVICE = 'cuda'
EMB_DIM = 256

MIN_LENGTH = 8

TRAIN_NUM_SAMPLES = 2000
TRAIN_MAX_SEQUENCE_LENGTH = 10

VAL_NUM_SAMPLES = 200
VAL_MAX_SEQUENCE_LENGTH = 20

N_STACK = 20
INITIAL_SHARPEN = 5.0

BATCH_SIZE = 32

GRAD_CLIP = None

# Symbols that go into palindrome
train_lang = 'a b c d e f g h i j'.split(' ')
# val_lang   = 'a b c d e f g h i j'.split(' ')
val_lang = 'k l m n o p q r s t'.split(' ')  # NOTE: these tokens are never trained in any example


##################################################
# ARCHITECTURES

##############################
# Transformer


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class PositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionEmbedding, self).__init__()
        self.pos_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        seq_len = x.size(0)
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        return x + self.pos_embedding(positions)

class AlibiPositionalBias(nn.Module):
    def __init__(self, num_heads, max_len=5000):
        super(AlibiPositionalBias, self).__init__()
        self.num_heads = num_heads
        self.max_len = max_len
        slopes = torch.Tensor(self._get_slopes(num_heads))
        bias = slopes.unsqueeze(1).unsqueeze(1) * torch.arange(max_len).unsqueeze(0).unsqueeze(0)
        self.register_buffer('bias', bias)

    def _get_slopes(self, n):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(n).is_integer():
            return get_slopes_power_of_2(n)
        else:
            closest_power_of_2 = 2 ** math.floor(math.log2(n))
            return get_slopes_power_of_2(closest_power_of_2) + self._get_slopes(2 * closest_power_of_2)[0::2][:n - closest_power_of_2]

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.bias[:, :, :seq_len]

# Add the relative positional encoding from Transformer-XL
class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(RelativePositionalEncoding, self).__init__()
        self.embedding = nn.Parameter(torch.randn(max_len, d_model))

    def forward(self, x):
        seq_len = x.size(0)
        return x + self.embedding[:seq_len]

class TransformerEncoder(nn.Module):
    def __init__(self, emb_dim, hidden_dim, num_layers, num_heads, tokenizer, pos_encoding='sine'):
        super(TransformerEncoder, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        if pos_encoding == 'sine':
            self.pos_encoder = PositionalEncoding(emb_dim)
        elif pos_encoding == 'learned':
            self.pos_encoder = PositionEmbedding(emb_dim)
        elif pos_encoding == 'alibi':
            self.pos_encoder = AlibiPositionalBias(num_heads)
        elif pos_encoding == 'transformerxl':
            self.pos_encoder = RelativePositionalEncoding(emb_dim)
        else:
            raise ValueError(f"Unsupported position encoding: {pos_encoding}")

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        x = self.embeddings(x_ids)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)
        outputs = F.softmax(x, dim=-1)
        return outputs


##############################
# LSTM

class LSTM(nn.Module):
    def __init__(self, emb_dim, hidden_dim, tokenizer):
        super(LSTM, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.lc1 = nn.LSTMCell(emb_dim, hidden_dim)
        self.lc2 = nn.LSTMCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        B = x_ids.size(0)
        x = self.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        c1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1, c1 = self.lc1(x[:, i, :], (h1, c1))
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1)  # [batch, seq, dim]
        h2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        c2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2, c2 = self.lc2(h1s[:, i], (h2, c2))
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


##############################
# RNN

class RNN(nn.Module):
    def __init__(self, emb_dim, hidden_dim, tokenizer):
        super(RNN, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.emb_dim = emb_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.lc1 = nn.RNNCell(emb_dim, hidden_dim)
        self.lc2 = nn.RNNCell(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, self.vocab_size)

    def forward(self, x_ids):
        B = x_ids.size(0)
        x = self.embeddings(x_ids)
        # init
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        h1s = []
        for i in range(x.size(1)):
            h1 = self.lc1(x[:, i, :], h1)
            h1s.append(h1)
        h1s = torch.stack(h1s, dim=1)  # [batch, seq, dim]
        h2 = torch.zeros(B, self.hidden_dim).to(x.device)  # [batch, dim]
        outputs = []
        for i in range(x.size(1)):
            h2 = self.lc2(h1s[:, i], h2)
            output = self.fc(h2)
            output = F.softmax(output, dim=1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=1)
        return outputs


##############################
# RNNStack

class RNNStack(nn.Module):
    def __init__(self, emb_dim, hidden_dim, tokenizer):
        super(RNNStack, self).__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        self.vocab_size = tokenizer.get_vocab_size()
        self.embeddings = nn.Embedding(self.vocab_size, emb_dim)

        self.rnn_i = nn.Linear(emb_dim, hidden_dim, bias=False)
        self.rnn_h = nn.Linear(hidden_dim, hidden_dim, bias=False)

        ##########
        # Outputs

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, self.vocab_size, bias=False),
            nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

        self.choose = nn.Sequential(
            nn.Linear(hidden_dim * 2, 2, bias=False),
            # nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
            GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

        ##########
        # Stack

        self.n_stack = N_STACK
        self.initial_sharpen = INITIAL_SHARPEN
        self.zero_offset = 0
        self.stack_vec_size = emb_dim
        self.sharp = nn.Parameter(torch.tensor([self.initial_sharpen]))

        self.ops = nn.Sequential(
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim * 2, 3, bias=False),  # [hidden, push+pop+nop]
            # nn.Softmax(dim=1)
            # GumbelSoftmax(dim=1, temperature=1.0, hard=True)
            GumbelSoftmax(dim=1, temperature=1.0, hard=False)
        )

    def forward(self, x_ids, debug=False):
        x = self.embeddings(x_ids)
        batch_size, device, dtype = x.size(0), x.device, x.dtype
        # init
        ss = S.initialize(self.emb_dim, self.n_stack, batch_size,
                          self.zero_offset + 1e-3, device, dtype=dtype)
        h = torch.zeros(x.size(0), self.hidden_dim).to(x.device)
        outputs = []
        if debug:
            ops_trace = []
        for i in range(x.size(1)):
            ri = self.rnn_i(F.normalize(x[:, i], dim=1))
            rh = self.rnn_h(h)
            h = (ri + rh).tanh()

            hmod = 1 - h.relu()
            hmod = torch.cat([hmod, hmod], dim=1)

            # Stack
            ops = self.ops(hmod)
            if debug:
                ops_trace.append(ops)
            push, pop, nop = [o.squeeze(1)
                              for o in torch.chunk(ops, chunks=3, dim=-1)]
            ss, pop_val = S.push_pop_nop(ss, self.sharp,
                                         push, pop, nop, x[:, i])

            # Outputs
            semantic = self.fc(hmod)
            # OPTIM: will this be memory hog?
            syntactic = torch.cosine_similarity(
                pop_val.unsqueeze(1),  # [B, 1, D]
                self.embeddings.weight.unsqueeze(0),  # [1, V, D]
                dim=2)  # [B, V]
            choice = self.choose(hmod)
            output = torch.einsum('bc, bcv -> bv',
                                  choice,
                                  torch.stack([semantic, syntactic], dim=1))
            outputs.append(output)

        outputs = torch.stack(outputs, dim=1)
        if debug:
            ops_trace = torch.stack(ops_trace, dim=1)
            debug_out = {'ops_trace': ops_trace}
            return outputs, debug_out
        else:
            return outputs



##################################################
# DATA

BOS_SYMBOL = '^'
REFLECT_SYMBOL = '|'
PAUSE_SYMBOL = '.'

def palindrome(num_samples,
               min_length,
               max_length,
               lang) -> Dict[str, List[str]]:
    data = []
    for _ in range(num_samples):
        length = random.randint(min_length, max_length)
        hlength = length // 2
        seed = [random.choice(lang) for _ in range(hlength)]
        # add pauses to inputs and outputs
        inputs = [BOS_SYMBOL] + seed + [REFLECT_SYMBOL] + [PAUSE_SYMBOL] * hlength
        outputs = [PAUSE_SYMBOL] * (hlength + 2) + seed[::-1]
        # convert all symbols to str
        inputs = list(map(str, inputs))
        outputs = list(map(str, outputs))
        data.append({
            'inputs': inputs,
            'outputs': outputs,
        })
    return data

train_raw = palindrome(TRAIN_NUM_SAMPLES, MIN_LENGTH,
                       TRAIN_MAX_SEQUENCE_LENGTH, lang=train_lang)
train_raw = sorted(train_raw, key=lambda x: len(x['inputs']))
# >>> train_raw[0]
#   {'inputs': ['^', 'a', 'b', 'c', 'd', '|', '.', '.', '.', '.'],
#   'outputs': ['.', '.', '.', '.', '.', '.', 'd', 'c', 'b', 'a']}

val_raw = palindrome(VAL_NUM_SAMPLES, MIN_LENGTH,
                     VAL_MAX_SEQUENCE_LENGTH, lang=val_lang)
val_raw = sorted(val_raw, key=lambda x: len(x['inputs']))

tokenizer, (train_dl, val_dl) = build_tokenizer_dataloader(
    [train_raw, val_raw],
    data_keys=['inputs', 'outputs'],
    batch_size=BATCH_SIZE)


# dataloader_info(train_dl, tokenizer)





##################################################
# Training

##########
# Setup

def train_model(config, tokenizer, train_dl, val_dl):
    seed = config['seed']
    torch.manual_seed(seed)
    random.seed(seed)

    # ##########
    # # Prep Data
    # train_dl = ray.train.get_dataset_shard("train")
    # val_dl = ray.train.get_dataset_shard("val")

    ##########
    # Prep Model
    if config['model_type'] == 'RNNStack':
        model = RNNStack(config['emb_dim'],
                         config['hidden_dim'],
                         tokenizer=tokenizer)
    elif config['model_type'] == 'RNN':
        model = RNN(config['emb_dim'],
                    config['hidden_dim'],
                    tokenizer=tokenizer)
    elif config['model_type'] == 'LSTM':
        model = LSTM(config['emb_dim'],
                     config['hidden_dim'],
                     tokenizer=tokenizer)

    model.to(DEVICE)

    # skip training embeddings
    no_trains = [model.embeddings]
    for no_train in no_trains:
        for p in no_train.parameters():
            p.requires_grad = False
    # print_model_info(model)

    params = [x for x in model.parameters() if x.requires_grad]
    optimizer = optim.Adam(params, lr=config['lr'], weight_decay=0)

    for epoch in range(config['num_epochs']):
        # Training
        tloss, tacc, _ = run_epoch(
            model, train_dl, optimizer, 'train', DEVICE, GRAD_CLIP,
            check_accuracy=True)

        # Validation
        vloss, vacc, _ = run_epoch(
            model, val_dl, None, 'eval', DEVICE,
            check_accuracy=True)

        ray.train.report(dict(epoch=epoch,
                              train_loss=tloss,
                              train_accuracy=tacc,
                              val_loss=vloss,
                              val_accuracy=vacc))


##########
# Run

LR = tune.grid_search([1e-3, 1e-2])
SEED = tune.grid_search([1, 2, 3])
NUM_EPOCHS = 10
HIDDEN_DIM = 32

RNN_config = {
    'model_type': 'RNN',
    'lr': LR,
    'emb_dim': EMB_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_epochs': NUM_EPOCHS,
    'seed': SEED,
}

LSTM_config = {
    'model_type': 'LSTM',
    'lr': LR,
    'emb_dim': EMB_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_epochs': NUM_EPOCHS,
    'seed': SEED,
}

RNNStack_config = {
    'model_type': 'RNNStack',
    'lr': LR,
    'emb_dim': EMB_DIM,
    'hidden_dim': HIDDEN_DIM,
    'num_epochs': NUM_EPOCHS,
    'seed': SEED,
}


all_configs = [
    RNN_config,
    LSTM_config,
    RNNStack_config,
]

def run_config(config):
    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_model,
                                 tokenizer=tokenizer,
                                 train_dl=train_dl,
                                 val_dl=val_dl),
            resources={"cpu": 2, "gpu": 2}),
        param_space=config,

        # num_samples=1, bc I explicitly set random seeds
        tune_config=tune.TuneConfig(num_samples=1),
    )

    results = tuner.fit()
    return results


for config in all_configs:
    config['results'] = run_config(config)


##################################################
# Visualization

def plot_experiments(all_configs, plot_confidence_bands=False):
    '''Plot train_loss, val_loss, train_accuracy, and val_accuracy for multiple
    different experiments.

    This finds the run with the top val_accuracy at any epoch, and plots its metrics.

    Also, optionally plot confidence bands across all all runs within an experiment.

    '''
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    axs = axs.flatten()

    color_scheme = [
        # dark    , light
        ('#1f77b4', '#aec7e8'),  # Blue
        ('#ff7f0e', '#ffbb78'),  # Orange
        ('#2ca02c', '#98df8a'),  # Green
        ('#d62728', '#ff9896'),  # Red
        ('#9467bd', '#c5b0d5'),  # Purple
        ('#8c564b', '#c49c94'),  # Brown
        ('#e377c2', '#f7b6d2'),  # Pink
        ('#7f7f7f', '#c7c7c7'),  # Gray
        ('#bcbd22', '#dbdb8d'),  # Olive
        ('#17becf', '#9edae5')   # Teal
    ]

    for i, config in enumerate(all_configs):
        model_type = config['model_type']
        train_color, val_color = color_scheme[i % len(color_scheme)]

        all_trials = config['results']

        # collect metrics for confidence bands
        all_train_loss = []
        all_val_loss = []
        all_train_accuracy = []
        all_val_accuracy = []
        all_epochs = []

        # keep trial run where top val_acc was achieved (at any epoch)
        max_val_accuracy = float('-inf')
        best_trial_index = None

        for j, trial in enumerate(all_trials):
            trial_df = trial.metrics_dataframe
            all_train_loss.append(trial_df['train_loss'])
            all_val_loss.append(trial_df['val_loss'])
            all_train_accuracy.append(trial_df['train_accuracy'])
            all_val_accuracy.append(trial_df['val_accuracy'])
            all_epochs.append(trial_df['epoch'])

            trial_max_val_accuracy = trial_df['val_accuracy'].max()
            if trial_max_val_accuracy > max_val_accuracy:
                max_val_accuracy = trial_max_val_accuracy
                best_trial_index = j

        # the trial that achieved the top val_acc at any epoch
        best_trial_df = all_trials[best_trial_index].metrics_dataframe
        best_train_loss = best_trial_df['train_loss']
        best_val_loss = best_trial_df['val_loss']
        best_train_accuracy = best_trial_df['train_accuracy']
        best_val_accuracy = best_trial_df['val_accuracy']

        for metric, ax, linestyle, (train_data, val_data, epochs) in [
            ('loss', axs[0], '-', (all_train_loss, all_val_loss, all_epochs)),
            ('accuracy', axs[1], '-', (all_train_accuracy, all_val_accuracy, all_epochs))
        ]:
            epochs = epochs[0]
            if plot_confidence_bands:
                train_mean = np.mean([data for data in train_data], axis=0)
                train_std = np.std([data for data in train_data], axis=0)
                val_mean = np.mean([data for data in val_data], axis=0)
                val_std = np.std([data for data in val_data], axis=0)

                ax.fill_between(epochs, train_mean - train_std, train_mean + train_std, color=train_color, alpha=0.2)
                ax.fill_between(epochs, val_mean - val_std, val_mean + val_std, color=val_color, alpha=0.2)

            ax.plot(epochs, best_train_loss if metric == 'loss' else best_train_accuracy, color=train_color, linestyle=linestyle, label=f"{model_type} - Best Train {metric.capitalize()}")
            ax.plot(epochs, best_val_loss if metric == 'loss' else best_val_accuracy, color=val_color, linestyle='--', label=f"{model_type} - Best Val {metric.capitalize()}")

            ax.set_xlabel("Epoch")
            ax.set_ylabel(metric.capitalize())
            ax.legend()

    plt.tight_layout()
    plt.show()

# Call the plotting function after running the configurations
plot_experiments(all_configs, plot_confidence_bands=False)
