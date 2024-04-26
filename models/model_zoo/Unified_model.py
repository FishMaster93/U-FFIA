import torch
from torch import nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class CrossAttention(nn.Module):
#     def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
#         super().__init__()
#         inner_dim = dim_head * heads
#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_q = nn.Linear(dim, inner_dim, bias=False)
#         self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         )

#     def forward(self, x, context):
#         q = self.to_q(x)
#         k, v = self.to_kv(context).chunk(2, dim=-1)

#         q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
#         k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
#         v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')

#         return self.to_out(out)
    
# class Attention(nn.Module):
#     def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
#         super().__init__()
#         inner_dim = dim_head *  heads
#         project_out = not (heads == 1 and dim_head == dim)

#         self.heads = heads
#         self.scale = dim_head ** -0.5

#         self.attend = nn.Softmax(dim = -1)
#         self.dropout = nn.Dropout(dropout)

#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

#     def forward(self, x):
#         qkv = self.to_qkv(x).chunk(3, dim = -1)
#         q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

#         attn = self.attend(dots)
#         attn = self.dropout(attn)

#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b h n d -> b n (h d)')
#         return self.to_out(out)



class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., cross_attention=False):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.cross_attention = cross_attention

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        if self.cross_attention:
            self.to_q = nn.Linear(dim, inner_dim, bias=False)
            self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        else:
            self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None):
        if self.cross_attention and context is not None:
            return self._forward_cross_attention(x, context)
        else:
            return self._forward_self_attention(x)

    def _forward_self_attention(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

    def _forward_cross_attention(self, x, context):
        q = self.to_q(x)
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b m (h d) -> b h m d', h=self.heads)
        v = rearrange(v, 'b m (h d) -> b h m d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)
    

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., cross_attention=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout, cross_attention=cross_attention)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))
        self.cross_attention = cross_attention

    def forward(self, x, context=None):
        for attn, ff in self.layers:
            if self.cross_attention and context is not None:
                x = attn(x, context=context) + x
            else:
                x = attn(x) + x
            x = ff(x) + x
        return x
    

class Unified_model(nn.Module):
    def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size,
                 num_classes, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0., cross_attention=False):
        super().__init__()

        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(image_patch_size)
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
        assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

        num_patches_v = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
        patch_dim_v = 3 * patch_height * patch_width * frame_patch_size
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding_v = nn.Sequential(
            Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
                      pf=frame_patch_size),
            nn.LayerNorm(patch_dim_v),
            nn.Linear(patch_dim_v, dim),
            nn.LayerNorm(dim),
        )

        self.to_patch_embedding_a = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, dim),
            nn.LayerNorm(dim),
        )
        num_patches_av = 8 + num_patches_v

        self.pos_embedding_v = nn.Parameter(torch.randn(1, num_patches_v + 1, dim))
        self.pos_embedding_a = nn.Parameter(torch.randn(1, 8 + 1, dim))
        self.pos_embedding_av = nn.Parameter(torch.randn(1, num_patches_av + 1, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, cross_attention=cross_attention)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head_a = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.mlp_head_v = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        self.mlp_head_av = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, audio, video, modality):
        if modality == 'v' and audio is None:
            x = self.to_patch_embedding_v(video)
            b, n, _ = x.shape

            cls_tokens_v = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens_v, x), dim=1)
            x += self.pos_embedding_v[:, :(n + 1)]

            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head_v(x)

        elif modality == 'a' and video is None:
            x = self.to_patch_embedding_a(audio)
            b, n, _ = x.shape
            cls_tokens_a = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
            x = torch.cat((cls_tokens_a, x), dim=1)
            x += self.pos_embedding_a[:, :(n + 1)]

            x = self.dropout(x)
            x = self.transformer(x)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head_a(x)

        elif modality == 'av':
            x_v = self.to_patch_embedding_v(video)
            x_a = self.to_patch_embedding_a(audio)
            b_v, n_v, _ = x_v.shape
            b_a, n_a, _ = x_a.shape

            cls_tokens_av_v = repeat(self.cls_token, '1 1 d -> b 1 d', b=b_v)
            cls_tokens_av_a = repeat(self.cls_token, '1 1 d -> b 1 d', b=b_a)
            x_v = torch.cat((cls_tokens_av_v, x_v), dim=1)
            x_a = torch.cat((cls_tokens_av_a, x_a), dim=1)

            x_v += self.pos_embedding_av[:, :(n_v + 1)]
            x_a += self.pos_embedding_av[:, :(n_a + 1)]

            x_v = self.dropout(x_v)
            x_a = self.dropout(x_a)

            x = self.transformer(x_v, x_a)
            x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head_av(x)
    
# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
#             ]))

#     def forward(self, x):
#         for attn, ff in self.layers:
#             x = attn(x) + x
#             x = ff(x) + x
#         return x


# class Unified_model(nn.Module):
#     def __init__(self, *, audio_size, image_size, image_patch_size, frames, frame_patch_size,
#                  num_classes, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()

#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(image_patch_size)
#         audio_height, audio_width = pair(audio_size)
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#         assert audio_height % patch_height == 0 and audio_width % patch_width == 0, 'Audio dimensions must be divisible by the patch size'
#         assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

#         num_patches_v = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
#         patch_dim_v = 3 * patch_height * patch_width * frame_patch_size
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding_v = nn.Sequential(
#             Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
#                       pf=frame_patch_size),
#             nn.LayerNorm(patch_dim_v),
#             nn.Linear(patch_dim_v, dim),
#             nn.LayerNorm(dim),
#         )

#         num_patches_a = (audio_height // patch_height) * (audio_width // patch_width)
#         patch_dim_a = 1 * patch_height * patch_width
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
#         self.to_patch_embedding_a = nn.Sequential(
#             Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
#             nn.LayerNorm(patch_dim_a),
#             nn.Linear(patch_dim_a, dim),
#             nn.LayerNorm(dim),
#         )
#         num_patches_av = num_patches_a + num_patches_v

#         self.pos_embedding_v = nn.Parameter(torch.randn(1, num_patches_v + 1, dim))
#         self.pos_embedding_a = nn.Parameter(torch.randn(1, num_patches_a + 1, dim))
#         self.pos_embedding_av = nn.Parameter(torch.randn(1, num_patches_av + 1, dim))

#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head_a = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.mlp_head_v = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.mlp_head_av = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, audio, video, modality):
#         if modality == 'v' and audio is None:
#             x = self.to_patch_embedding_v(video)
#             b, n, _ = x.shape

#             cls_tokens_v = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#             x = torch.cat((cls_tokens_v, x), dim=1)
#             x += self.pos_embedding_v[:, :(n + 1)]

#         elif modality == 'a' and video is None:
#             x = self.to_patch_embedding_a(audio)
#             b, n, _ = x.shape
#             cls_tokens_a = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#             x = torch.cat((cls_tokens_a, x), dim=1)
#             x += self.pos_embedding_a[:, :(n + 1)]

#         elif modality == 'av':
#             x = torch.cat((self.to_patch_embedding_v(video), self.to_patch_embedding_a(audio)), dim=1)
#             b, n, _ = x.shape

#             cls_tokens_av = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#             x = torch.cat((cls_tokens_av, x), dim=1)
#             x += self.pos_embedding_av[:, :(n + 1)]

#         x = self.dropout(x)
#         x = self.transformer(x)
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#         x = self.to_latent(x)
#         if modality == 'v' and audio is None:
#             return self.mlp_head_v(x)
#         elif modality == 'a' and video is None:
#             return self.mlp_head_a(x)
#         else:
#             return self.mlp_head_av(x)


# class Unified_model(nn.Module):
#     def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size,
#                  num_classes, dim, depth, heads, mlp_dim, pool='cls', dim_head=64, dropout=0., emb_dropout=0.):
#         super().__init__()

#         image_height, image_width = pair(image_size)
#         patch_height, patch_width = pair(image_patch_size)
#         assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
#         assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'

#         num_patches_v = (image_height // patch_height) * (image_width // patch_width) * (frames // frame_patch_size)
#         patch_dim_v = 3 * patch_height * patch_width * frame_patch_size
#         assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

#         self.to_patch_embedding_v = nn.Sequential(
#             Rearrange('b c (f pf) (h p1) (w p2) -> b (f h w) (p1 p2 pf c)', p1=patch_height, p2=patch_width,
#                       pf=frame_patch_size),
#             nn.LayerNorm(patch_dim_v),
#             nn.Linear(patch_dim_v, dim),
#             nn.LayerNorm(dim),
#         )


#         self.to_patch_embedding_a = nn.Sequential(
#             nn.LayerNorm(512),
#             nn.Linear(512, dim),
#             nn.LayerNorm(dim),
#         )
#         num_patches_av = 8 + num_patches_v

#         self.pos_embedding_v = nn.Parameter(torch.randn(1, num_patches_v + 1, dim))
#         self.pos_embedding_a = nn.Parameter(torch.randn(1, 8 + 1, dim))
#         self.pos_embedding_av = nn.Parameter(torch.randn(1, num_patches_av + 1, dim))

#         self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
#         self.dropout = nn.Dropout(emb_dropout)

#         self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

#         self.pool = pool
#         self.to_latent = nn.Identity()

#         self.mlp_head_a = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.mlp_head_v = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )
#         self.mlp_head_av = nn.Sequential(
#             nn.LayerNorm(dim),
#             nn.Linear(dim, num_classes)
#         )

#     def forward(self, audio, video, modality):
#         if modality == 'v' and audio is None:
#             x = self.to_patch_embedding_v(video)
#             b, n, _ = x.shape

#             cls_tokens_v = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#             x = torch.cat((cls_tokens_v, x), dim=1)
#             x += self.pos_embedding_v[:, :(n + 1)]

#         elif modality == 'a' and video is None:
#             x = self.to_patch_embedding_a(audio)
#             b, n, _ = x.shape
#             cls_tokens_a = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
#             x = torch.cat((cls_tokens_a, x), dim=1)
#             x += self.pos_embedding_a[:, :(n + 1)]

#         elif modality == 'av':
#             x = torch.cat((self.to_patch_embedding_v(video), self.to_patch_embedding_a(audio)), dim=1)
#             b, n, _ = x.shape

#             cls_tokens_av = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
#             x = torch.cat((cls_tokens_av, x), dim=1)
#             x += self.pos_embedding_av[:, :(n + 1)]

#         x = self.dropout(x)
#         x = self.transformer(x)
#         x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
#         x = self.to_latent(x)
#         if modality == 'v' and audio is None:
#             return self.mlp_head_v(x)
#         elif modality == 'a' and video is None:
#             return self.mlp_head_a(x)
#         else:
#             return self.mlp_head_av(x)





def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = Unified_model(
            image_size=224,
            audio_size=128,
            image_patch_size=16,
            frames=10,
            frame_patch_size=2,
            num_classes=4,
            dim=1024,
            depth=4,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1).cuda()

    video = torch.randn(2, 3, 8, 224, 224).cuda()
    audio = torch.randn(2, 1, 128, 128).cuda()
    output = model(None, video, 'v')
    print(output.shape)
