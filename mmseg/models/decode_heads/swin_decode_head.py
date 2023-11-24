# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead
import torch.nn as nn
from mmcv.cnn import ConvModule


from mmseg.models.backbones.swin import SwinBlockSequence


class up_conv(nn.Module):
    """
    Up Convolution Block
    """

    def __init__(self, in_ch, out_ch):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
            nn.GroupNorm(num_channels=out_ch, num_groups=32),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


@MODELS.register_module()
class SwinTHead(BaseDecodeHead):
    def __init__(self,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 act_cfg=dict(type='GELU'),
                 swin_norm_cfg=dict(type='LN'),
                 with_cp=False, 
                 **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stage_list = []
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.swin_block3 = SwinBlockSequence(
                        embed_dims=self.in_channels[2+1],
                        num_heads=num_heads[2+1],
                        feedforward_channels=int(mlp_ratio * self.in_channels[2+1]),
                        depth=depths[2+1],
                        window_size=window_size,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:2+1]):sum(depths[:2+1 + 1])],
                        downsample=None,
                        act_cfg=act_cfg,
                        norm_cfg=swin_norm_cfg,
                        with_cp=with_cp,
                        init_cfg=None)
        self.up3 = up_conv(192, 192)
        self.swin_block2 = SwinBlockSequence(
                        embed_dims=self.in_channels[1+1],
                        num_heads=num_heads[1+1],
                        feedforward_channels=int(mlp_ratio * self.in_channels[1+1]),
                        depth=depths[1+1],
                        window_size=window_size,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:1+1]):sum(depths[:1+1 + 1])],
                        downsample=None,
                        act_cfg=act_cfg,
                        norm_cfg=swin_norm_cfg,
                        with_cp=with_cp,
                        init_cfg=None)
        self.up2 = up_conv(192*2, 192)
        self.swin_block1 = SwinBlockSequence(
                        embed_dims=self.in_channels[0+1],
                        num_heads=num_heads[0+1],
                        feedforward_channels=int(mlp_ratio * self.in_channels[0+1]),
                        depth=depths[0+1],
                        window_size=window_size,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        drop_rate=drop_rate,
                        attn_drop_rate=attn_drop_rate,
                        drop_path_rate=dpr[sum(depths[:0+1]):sum(depths[:0+1 + 1])],
                        downsample=None,
                        act_cfg=act_cfg,
                        norm_cfg=swin_norm_cfg,
                        with_cp=with_cp,
                        init_cfg=None)
        self.up1 = up_conv(192*2, 192)

        self.change_channels = nn.Sequential(ConvModule(
                        in_channels=256,
                        out_channels=192,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg),
                        ConvModule(
                        in_channels=256,
                        out_channels=192,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg),
                        ConvModule(
                        in_channels=256,
                        out_channels=192,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg),
                        ConvModule(
                        in_channels=256,
                        out_channels=192,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=act_cfg))

    def forward(self, inputs):
        inputs = [self.change_channels[i](inputs[i]) for i in range(len(inputs))]
        x0, x1, x2, x3 = inputs
        B, C, H, W = x3.shape
        
        x3 = x3.permute(0, 2, 3, 1).contiguous().view(B, -1, C)
        x3 = self.swin_block3(x3, (H, W))[0]
        x3 = x3.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        x3 = self.up3(x3)

        x2 = torch.cat([x2, x3], dim=1)
        x2 = x2.permute(0, 2, 3, 1).contiguous().view(B, -1, C*2)
        x2 = self.swin_block2(x2, (H*2, W*2))[0]
        x2 = x2.view(B, H*2, W*2, C*2).permute(0, 3, 1, 2).contiguous()
        x2 = self.up2(x2)

        x1 = torch.cat([x1, x2], dim=1)
        x1 = x1.permute(0, 2, 3, 1).contiguous().view(B, -1, C*2)
        x1 = self.swin_block1(x1, (H*4, W*4))[0]
        x1 = x1.view(B, H*4, W*4, C*2).permute(0, 3, 1, 2).contiguous()
        x1 = self.up1(x1)

        x0 = torch.cat([x0, x1], dim=1)

        output = self.cls_seg(x0)
        return output
