from typing import Mapping, Any, List
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.model_zoo import load_url

import torchvision
from torchvision.models.vgg import VGG, make_layers

vgg16_cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

class VGGEncoder(VGG):
    def __init__(self, in_channels, depth=5) -> None:
        super().__init__(make_layers(cfg=vgg16_cfg, batch_norm=False))

        self.in_channels = in_channels
        self.depth = depth

    def get_stage(self):
        stages = []
        stage_modules = []
        for module in self.features:
            if isinstance(module, nn.MaxPool2d):
                stages.append(nn.Sequential(*stage_modules))
                stage_modules = []

            stage_modules.append(module)
        stages.append(nn.Sequential(*stage_modules))
        return stages

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        stages = self.get_stage()

        features = []
        for i in range(self.depth + 1):
            x = stages[i](x)
            features.append(x)

        return features
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        # keys = list(state_dict.keys())
        # for k in keys:
        #     if k.startswith("classifier"):
        #         state_dict.pop(k, None)
        super().load_state_dict(state_dict, strict)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Convolutional block used in Encoder and Decoder block
        
        `[[Conv2d(3, 1)] -> [BatchNorm] -> [ReLU]] * 2`

        Args:
            in_channels (int): number of channels of input features or images
            out_channels (int): number of desired channels of output
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def get_out_channels(self):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_block(x)
    

class UNetDecoder(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """Decoder block

        `[interpolate('nearest') -> crop_and_concat -> ConvBlock]`

        Args:
            in_channels (int): number of channels of input
            out_channels (int): number of desired channels of output
        """
        super().__init__()
        self.conv_block = ConvBlock(in_channels + skip_channels, out_channels)

    def forward(self, x, skip_conn):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        
        x = torch.cat([skip_conn, x], dim=1)

        return self.conv_block(x)
    

class VGG_UNet(nn.Module):
    def __init__(self, in_channels: int = 3, seg_out_channels: int = 3, clf_out_channels: int = 1000, encoder_pretrained: bool = True) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.seg_out_channels = seg_out_channels
        self.clf_out_channels = clf_out_channels
        self.encoder_pretrained = encoder_pretrained

        self.encoder = VGGEncoder(self.in_channels)

        if self.encoder_pretrained:
            vgg_state_dict = load_url("https://download.pytorch.org/models/vgg16-397923af.pth")
            self.encoder.load_state_dict(state_dict=vgg_state_dict)

        if self.clf_out_channels != 1000:
            self.encoder.classifier[-1] = nn.Linear(in_features=4096, out_features=self.clf_out_channels)

        self.classifier_head = copy.deepcopy(self.encoder.classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        encoder_channels = (64, 128, 256, 512, 512, 512)
        decoder_channels = (256, 128, 64, 32, 16)

        encoder_channels = encoder_channels[::-1]
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])

        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        self.center = ConvBlock(head_channels, head_channels)

        blocks = [
            UNetDecoder(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]

        self.decoder = nn.ModuleList(blocks)

        self.segmentation_head = nn.Conv2d(in_channels=16, out_channels=self.seg_out_channels, kernel_size=3, padding='same')

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)

        clf_head = self.avgpool(features[-1])
        clf_head = torch.flatten(clf_head, 1)
        clf_head = self.classifier_head(clf_head)

        features = features[::-1]

        head = features[0]
        skips = features[1:]

        segment_head = self.center(head)

        for i, dec in enumerate(self.decoder):
            skip = skips[i] if i < len(skips) else None
            segment_head = dec(segment_head, skip)

        segment_head = self.segmentation_head(segment_head)

        return segment_head, clf_head