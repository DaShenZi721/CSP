import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import matplotlib.ticker as mticker
import os
import logging
from collections import defaultdict
import pickle
import wandb

from datetime import datetime

def create_logger(args):
    dataset_name = args['dataset_name']

    log_path = os.path.join(args['output_path'], dataset_name, "logs")
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    seed = args['seed']

    log_name = "{}_{}_{}.log".format(dataset_name, seed, time_str)
    log_file = os.path.join(log_path, log_name)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    print("=> creating log {}".format(log_file))
    header = "%(asctime)-15s %(message)s"
    logging.basicConfig(filename=str(log_file), format=header)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger("").addHandler(console)

    logger.info("---------------------Cfg is set as follow--------------------")
    for k, v in args.items():
        logger.info('%s:%s' % (k, str(v)))
    logger.info("-------------------------------------------------------------")
    return logger, log_file, log_name.split('.')[0]

def seed_everything(seed=666):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def save_checkpoint(model, args, save_checkpoint_path):
    state = {
        'args': dict(args),
        'state_dict': model.state_dict(),
    }
    path = os.path.join(save_checkpoint_path, 'best_model.pth')
    torch.save(state, path)


def load_state_dict(path, model):
    if isinstance(path, str):
        state = torch.load(path, map_location=lambda storage, loc: storage)
    else:
        state = path

    model_state_dict = model.state_dict()
    if 'state_dict' in state.keys():
        loaded_state_dict = state['state_dict']
    else:
        loaded_state_dict = state

    pretrained_state_dict = {}
    for param_name in loaded_state_dict.keys():
        new_param_name = param_name
        if new_param_name not in model_state_dict:
            print(f'Pretrained parameter "{param_name}" cannot be found in model parameters.')
        elif model_state_dict[new_param_name].shape != loaded_state_dict[param_name].shape:
            print(f'Pretrained parameter "{param_name}" '
                  f'of shape {loaded_state_dict[param_name].shape} does not match corresponding '
                  f'model parameter of shape {model_state_dict[new_param_name].shape}.')
        else:
            # print(f'Loading pretrained parameter "{param_name}".')
            pretrained_state_dict[new_param_name] = loaded_state_dict[param_name]
    # Load pretrained weights
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    
    return model
    
def visual_matrix(attn_weights, save_fig='.figs/p_haar.png'):
    # |attn_weights| : [(batch_size, n_heads, seq_len, seq_len)]
    # print(len(attn_weights))
    # print(attn_weights[0].shape) # [batch_size: 64, n_heads: 8, seq_len: 65, dim_head: 64]
    # 1/0
    part = [1,2,4,6]
    fig, axes = plt.subplots(nrows=1, ncols=len(part), figsize=(10, 3))
    # for i, attention_weight in enumerate(attn_weights):

    for i, ax in enumerate(axes.flat):
        idx = part[i]-1
        # attention_weight = F.one_hot(attention_weight).permute(0, 1, 3, 2, 4)
        # ax.set_axis_off()
        # print(torch.linalg.matrix_rank(attn_weights[i][0,0]))
        im = ax.matshow(attn_weights[idx][0,0])
        major_locator=MultipleLocator(20)
        ax.xaxis.set_major_locator(major_locator)
        ax.yaxis.set_major_locator(major_locator)
        
        ax.tick_params(axis='both', which='major', labelsize=17.5)
        ax.set_xlabel('Layer %d' % (idx+1), fontdict={'size':23})
        
    fig.subplots_adjust(bottom=0.0, top=1.0, left=0.05, right=0.91,
                    wspace=0.24, hspace=0.0)
    cb_ax = fig.add_axes([0.93, 0.12, 0.02, 0.76])
    locator = mticker.MultipleLocator(1.0)
    formatter = mticker.StrMethodFormatter('{x:.1f}')
    cb_ax.tick_params(labelsize=15)
    fig.colorbar(im, cax=cb_ax, ticks=locator, format=formatter)
    # fig.tight_layout()
    if save_fig:
        plt.savefig(save_fig, dpi=150)
    plt.show()

def visual_atten(attention_weights):
    # |attention_weights| : [epoch x (batch_size, n_heads, seq_len, seq_len)]
    for i in range(0, len(attention_weights), int(len(attention_weights) / 10)):
        col_sorted, col_indices = attention_weights[i][0, 0].sum(-2).sort(-1)
        x = list(range(len(col_sorted)))
        l = plt.plot(x, col_sorted, label=i)
    plt.xlabel('Sorted columns')
    plt.ylabel('Sum of coefficients')
    plt.legend()
    plt.show()
