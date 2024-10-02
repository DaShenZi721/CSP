import torch
import torch.nn.functional as F

config = {
    'load_checkpoint_path': '',
    'output_path': './output/',
    
    'wandb': False,
    'validate': True,
    'visual_attn': False,
    
    'gpu': 'cuda:0',
    'num_workers': 8,
    'seed': 0,
    
    'model': 'vit', # choices=['vit', 'ema', 'Haar']
    'attn': 'swd', #choices=['trans', 'swd']
    'len_sort_window': 2,
    
    'n_epochs': 100,
    'batch_size': 64, # 64
    'lr': 0.0001,
    
    'pool': 'mean', # choices=['cls', 'mean', 'max']
    'n_layers': 6,
    'n_heads': 8,
    'dim': 512, # 512
    'dim_head': 64,
    'mlp_dim': 512, # 512
    'emb_dropout': 0.1,
    'dropout': 0.1,
    
    'dataset_name': 'cifar10', # choices=['mnist', 'cifar10', 'cifar100', 'FakeData']
    'size': 32,
    'ps': 4,
    'num_classes': 10,
    'channels': 3,
    'padding': 4,
}
