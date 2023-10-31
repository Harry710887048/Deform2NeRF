import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from lib.config import cfg
from hash_encoding import HashEmbedder, SHEncoder

# Misc
img2mse = lambda x, y: torch.mean((x - y) ** 2)
mse2psnr = lambda x: -10. * torch.log(x) / torch.log(torch.Tensor([10.]))
to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)


# Positional encoding (section 5.1)
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


def get_embedder(multires, args, i=0):
    if i == -1:
        return nn.Identity(), 3
    elif i == 0:
        embed_kwargs = {
            'include_input': True,
            'input_dims': 3,
            'max_freq_log2': multires - 1,
            'num_freqs': multires,
            'log_sampling': True,
            'periodic_fns': [torch.sin, torch.cos],
        }  # 位置编码

        embedder_obj = Embedder(**embed_kwargs)
        embed = lambda x, eo=embedder_obj: eo.embed(x)
        out_dim = embedder_obj.out_dim
    elif i == 1:
        # args.bounding_box为3d边界框，由其最小顶点和最大顶点进行表示
        # args.log2_hashmap_size为每层分辨率的哈希表大小，论文中取2^19
        # args.finest_res为最大分辨率
        embed = HashEmbedder(bounding_box=args.bounding_box, log2_hashmap_size=args.log2_hashmap_size,
                             finest_resolution=args.finest_res)  # 哈希编码
        out_dim = embed.out_dim
    elif i == 2:
        embed = SHEncoder()
        out_dim = embed.out_dim
    return embed, out_dim


# xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res, args='', i=1)  # arg表示hash编码的参数
# view_embedder, view_dim = get_embedder(cfg.view_res, args='', i=2)
