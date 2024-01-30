# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import math
from functools import partial

import torch
import torch.nn as nn

from vit_local import PatchEmbed, Block

from util.pos_embed import get_2d_sincos_pos_embed

def weight_reinitialization(m):
    if isinstance(m, nn.Linear):
        # we use xavier_uniform following official JAX ViT:
        torch.nn.init.xavier_uniform_(m.weight)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)

class DiversityLoss(nn.Module):
    def __init__(self, distance_metric='euclidean', scale_factor=0.1):
        super(DiversityLoss, self).__init__()
        self.distance_metric = distance_metric
        self.scale_factor = scale_factor

    def forward(self, outputs):
        n = outputs.size()[0]  # size of batch
        if n == 1:
            return 0.0
        # compute distance matrix between outputs
        if self.distance_metric == 'euclidean':
            distances = torch.cdist(outputs, outputs, p=2)
        else:
            raise ValueError('Unsupported distance metric')
        # compute diversity loss
        loss = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                loss += torch.exp(-distances[i, j] * self.scale_factor)
        # normalize loss by batch size
        loss /= n * (n - 1) / 2
        return loss

class MaskingNet(nn.Module):
    """ Masking network """
    def __init__(self, num_tokens, embed_dim=256, depth=5, num_heads=4, mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])

        self.mlp_head = nn.Sequential(
            norm_layer(embed_dim),
            nn.Linear(embed_dim, num_tokens),
            nn.Sigmoid()
        )

        self.initialize_weights()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], 1, cls_token=True)[:self.pos_embed.shape[1]]
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def forward(self, x):
        cls_token = self.cls_token + self.pos_embed
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.mlp_head(x[:, 0])
        return x


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=512, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, lambda_gauss=1.0,
                 lambda_kl=0.01, lambda_diversity = 2.0):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE masking net specifics
        self.masking_net = MaskingNet(num_tokens=num_patches, embed_dim=embed_dim)
        self.alpha = -1 / (0.12 * 0.12 * 2)
        self.beta = 1 / (0.12 * math.sqrt(2 * math.pi))
        self.diversity_loss = DiversityLoss()

        self.lambda_gauss = lambda_gauss
        self.lambda_kl = lambda_kl
        self.lambda_diversity = lambda_diversity
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def kl_divergence(self, p, q):
        p = torch.as_tensor(p, dtype=torch.float32)
        q = torch.as_tensor(q, dtype=torch.float32)
        return torch.sum(p * torch.log(p / q), dim=-1)
        
    def freeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = False

    def unfreeze_maskingnet(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                param.requires_grad = True

    def freeze_backbone(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                continue
            if "pos_embed" in name:
                continue

            param.requires_grad = False

    def unfreeze_backbone(self):
        for name, param in self.named_parameters():
            if "masking_net" in name:
                continue
            if "pos_embed" in name:
                continue
            param.requires_grad = True

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def curriculum_masking(self, x, mask_ratio, train_mask):
        """
        Perform per-sample masking.
        Masks are generated by "masking_net" facilitating curriculum learning
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        mask_embedding = self.masking_net(x)

        if train_mask:
            ids_shuffle = torch.argsort(mask_embedding, dim=1, descending=True)  # descend: small is remove, large is keep
            ids_restore = torch.argsort(ids_shuffle, dim=1)

            x = x * mask_embedding.unsqueeze(-1)
            x = torch.gather(x, dim=1, index=ids_shuffle.unsqueeze(-1).repeat(1, 1, D))
            return x, mask_embedding, ids_restore

        # sort masking output for each sample
        ids_shuffle = torch.argsort(mask_embedding, dim=1, descending=True)  # descend: small is remove, large is keep
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


    def forward_encoder(self, x, mask_ratio, train_mask):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.curriculum_masking(x, mask_ratio, train_mask)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_loss_mask(self, imgs, pred, masking_output, mask_ratio, ids_restore):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2

        mm = (1-masking_output)
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss_reconst = (loss * mm).sum() / mm.sum()  # mean loss on removed patches

        ## Gaussian loss: converges the output towards 0s and 1s
        loss_gauss = self.lambda_gauss * (torch.exp(self.alpha*(masking_output - 0.5)**2) / self.beta).mean()
        ## KL Divergence loss ensure the fixed masking ratio
        loss_kl = self.lambda_kl * (self.kl_divergence(masking_output.mean(-1), 0.25) + self.kl_divergence((1-masking_output).mean(-1), 0.75))
        ## Diversity loss to prevent converging module to generate fixed set of mask for different samples
        loss_diversity = self.lambda_diversity * self.diversity_loss(masking_output)

        return loss_reconst, loss_gauss, loss_kl, loss_diversity

    def forward(self, imgs, mask_ratio=0.75, train_mask=False):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, train_mask)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]

        if train_mask:
            loss_recon, loss_gauss, loss_kl, loss_diversity = self.forward_loss_mask(imgs, pred, mask, mask_ratio, ids_restore)
            return loss_recon, loss_gauss, loss_kl, loss_diversity
        else:
            loss = self.forward_loss(imgs, pred, mask)
            return loss

def mae_vit_base_patch8_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=8, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch8 = mae_vit_base_patch8_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
