import torch
from torch import nn
import torch.nn.functional as F
from vit_pytorch.sinkhorn import SinkhornDistance
from vit_pytorch.swd import *
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class Sorting(nn.Module):
    def __init__(self, dim: int = -1):
        super(Sorting, self).__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed forward step of x
        x: a tensor with arbitrary size
        """
        x, _ = torch.sort(x, dim=self.dim)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, args):
        super().__init__()
        dim = args['dim']
        hidden_dim = args['mlp_dim']
        dropout = args['dropout']
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

perm = torch.randperm(64)

class Attention(nn.Module):
    def __init__(self, args, layer_idx=None):
        super().__init__()
        dim = args['dim']
        dim_head = args['dim_head']
        dropout = args['dropout']
        self.heads = args['n_heads']
        self.attn = args['attn']
        
        inner_dim = dim_head *  self.heads
        project_out = not (self.heads == 1 and dim_head == dim)

        self.scale = dim_head ** -0.5
        self.swd = SWD21(args, layer_idx=layer_idx)
        self.sink = SinkhornDistance(eps=1, max_iter=3)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        if self.attn == 'swd':
            self.to_qkv = nn.Linear(dim, inner_dim, bias = False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()
            
    def forward(self, x):
        attn_map = None
        if self.attn == 'trans':
            qkv = self.to_qkv(x).chunk(3, dim = -1)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale            
            attn = self.attend(dots)
            
            # sinkformer
            # dots_former_shape = dots.shape
            # dots = dots.view(-1, dots_former_shape[2], dots_former_shape[3])
            # attn = self.sink(dots)[0]
            # attn = attn * attn.shape[-1]
            # attn = attn.view(dots_former_shape)
            
            # row max binary
            # attn_max_idx = torch.argmax(attn, dim=-1).unsqueeze(dim=-1)
            # attn = torch.zeros_like(attn).scatter_(-1, attn_max_idx, 1.)
            # attn_map = attn
            
            attn = self.dropout(attn)
            out = torch.matmul(attn, v)
            attn_map = out
            
            out = rearrange(out, 'b h n d -> b n (h d)')
        elif self.attn == 'swd':
            v = self.to_qkv(x)
            out = self.swd(v)
            attn_map = out
        
        U, S, Vh = torch.linalg.svd(out)
        print(S[0])
            
        return self.to_out(out), attn_map

class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([])
        for idx, _ in enumerate(range(args['n_layers'])):
            self.layers.append(nn.ModuleList([
                PreNorm(args['dim'], Attention(args, layer_idx=idx)),
                PreNorm(args['dim'], FeedForward(args))
            ]))
            
    def forward(self, x):
        attn_weights = []
        for idx, (attn, ff) in enumerate(self.layers):           
            attn_x, attn_map = attn(x)
            x = attn_x + x
            x = ff(x) + x            
            attn_weights.append(attn_map.detach().clone().cpu())
        1/0
            
        return x, attn_weights


class ViT(nn.Module):    
    def __init__(self, args):
        super().__init__()
        image_height, image_width = pair(args['size'])
        patch_height, patch_width = pair(args['ps'])

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = args['channels'] * patch_height * patch_width

        dim = args['dim']
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(args['emb_dropout'])

        self.transformer = Transformer(args)

        self.pool = args['pool']
        assert self.pool in {'cls', 'mean', 'max'}, 'pool type must be either cls (cls token), mean (mean pooling) or max'
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, args['num_classes'])
        )
        
        self.is_adv = True if 'is_adv' in args and args['is_adv'] else False

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        if self.pool == 'cls':
            cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
            x = torch.cat((cls_tokens, x), dim=1)
            x += self.pos_embedding[:, :(n + 1)]
        else:
            x += self.pos_embedding[:, :n]
            
        x = self.dropout(x)

        trans_x, attn_weights = self.transformer(x)
        x = trans_x

        if self.pool == 'mean':
            x = x.mean(dim = 1)
        elif self.pool == 'max':
            x, _ = x.max(dim = 1)
        else:
            x = x[:, 0]

        x = self.to_latent(x)
        x = self.mlp_head(x)
        if self.is_adv:
            return x
        else:
            return x, attn_weights
