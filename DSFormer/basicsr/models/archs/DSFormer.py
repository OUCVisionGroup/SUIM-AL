import torch.nn.functional as F
import typing as t
import numpy as np
import torch.nn as nn
from einops import rearrange
import math
import warnings
from wandb.wandb_torch import torch
from torch.nn.init import _calculate_fan_in_and_fan_out
from PIL import Image


class Illumination_Estimator(nn.Module):
    def __init__(self, n_fea_middle=40, n_fea_in=4, n_fea_out=3):
        super().__init__()

        # 物理引导预处理
        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)

        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        # V通道处理分支
    def rgb_to_lab(self, rgb_img):
            """
            将 RGB 图像转换为 LAB 图像
            输入：RGB 图像 (B, C, H, W)
            输出：LAB 图像 (B, C, H, W)
            """
            batch_size, _, height, width = rgb_img.size()
            rgb_img = rgb_img.permute(0, 2, 3, 1).cpu()  # 转为 (B, H, W, C) 并转回 CPU

            lab_img = torch.zeros_like(rgb_img).float()  # 初始化空的 LAB 图像

            # 使用 PIL 转换每一张图像
            for b in range(batch_size):
                # 将 RGB 图像转为 PIL 格式
                pil_image = Image.fromarray((rgb_img[b].detach().cpu().numpy() * 255).astype('uint8'))

                # 转换为 LAB 色彩空间
                lab_image = pil_image.convert('LAB')

                # 将 LAB 图像转换为 numpy 数组
                lab_array = torch.tensor(np.array(lab_image)).float()

                # 归一化到 [0, 1] 范围
                lab_array = lab_array / 255.0

                # 存储转换后的图像
                lab_img[b] = lab_array

            # 转换回原设备
            lab_img = lab_img.to(rgb_img.device)

            # 转换维度为 (B, C, H, W)
            return lab_img.permute(0, 3, 1, 2)

    def forward(self, img):
        # 物理衰减估计
        lab_img = self.rgb_to_lab(img)
        device = img.device  # 获取设备（GPU 或 CPU）
                # 提取 L 通道（亮度通道）
        L = lab_img[:, 0:1, :, :]  # L 通道，亮度信息
        L = L.to(img.device)
        # 输入增强：拼接衰减系数图
        mean_c = img.mean(dim=1).unsqueeze(1)
        input = torch.cat([L , img], dim=1)
        # 多尺度特征提取
        x_1 = self.conv1(input)
        # 散射感知处理
        spatial_feat = self.depth_conv(x_1)
        # 频域特征融合
        illu_map = self.conv2(spatial_feat)
        return spatial_feat, illu_map ,img # 保持原始输出格式


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            SwiGLU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            SwiGLU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            SwiGLU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class SwiGLU(nn.Module):
    def forward(self, x):
        return x * F.sigmoid(x)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

class Denoiser(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        self.embedding = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)

        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        fea = self.embedding(x)

        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        fea = self.bottleneck(fea, illu_fea)

        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        out = self.mapping(fea) + x

        return out,fea


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat,
                                 level=level, num_blocks=num_blocks)
        # self.saliency_branch = SaliencyFPN(in_ch=3).cuda()

    def forward(self, img):
        illu_fea, illu_map, img = self.estimator(img)
        fire_img = img * illu_map
        # xiangjia = illu_map + img
        input_img = fire_img + img
        # L, AB = rgb_to_lab( input_img)
        output_img, en_fea = self.denoiser(input_img, illu_fea)
        output_img = (output_img + 1) / 3
        output_img = torch.clamp(output_img, 0, 1)  # 稳妥起见，保底一下
        # s_map, s_list = self.saliency_branch(img)
        return output_img, illu_map, input_img, img

#双分支
class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=40, stage=3, num_blocks=[1, 1, 1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage
        modules_body = [RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels,
                                                   n_feat=n_feat, level=2, num_blocks=num_blocks) for _ in range(stage)]
        self.body = nn.Sequential(*modules_body)
    def forward(self, x1, x2=None):
        if self.training and x2 is not None:  # 🔥 训练模式，用双分支
            color_img1, illu1, input1, img1 = self.body(x1)
            color_img2, illu2, input2, img2 = self.body(x2)
            return color_img1, color_img2, illu1, illu2, input1, input2, img1, img2
        else:  # 🔥 验证模式，只跑单分支
            color_img1, illu1, input1, img1= self.body(x1)
            return color_img1, illu1, input1, img1

