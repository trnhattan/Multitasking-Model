import torch
import torch.nn as nn
import torch.nn.functional as F

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
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Encoder block

        `[ConvBlock -> MaxPooling(2, 2)]`

        Args:
            in_channels (int): number of channels of input
            out_channels (int): number of desired channels of output
        """
        super().__init__()
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        """Decoder block

        `[ConvTranspose(2, 1) -> crop_and_concat -> ConvBlock]`

        Args:
            in_channels (int): number of channels of input
            out_channels (int): number of desired channels of output
        """
        super().__init__()

        self.conv_trans = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=1)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x, skip_conn):
        x = self.conv_trans(x)

        diffY = skip_conn.size()[2] - x.size()[2]
        diffX = skip_conn.size()[3] - x.size()[3]

        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip_conn, x], dim=1)

        return self.conv_block(x)

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        """U-Net

        Args:
            in_channels (int, optional): number of channels of input images.
                (Defaults: `3`)
            out_channels (int, optional): number of classes given in groundtruth.
                (Defaults: `3`)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.in_conv = ConvBlock(in_channels, 64)

        self.enc_1 = Encoder(64, 128)
        self.enc_2 = Encoder(128, 256)
        self.enc_3 = Encoder(256, 512)
        self.enc_4 = Encoder(512, 1024)

        self.dec_1 = Decoder(1024, 512)
        self.dec_2 = Decoder(512, 256)
        self.dec_3 = Decoder(256, 128)
        self.dec_4 = Decoder(128, 64)

        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.enc_1(x1)
        x3 = self.enc_2(x2)
        x4 = self.enc_3(x3)
        x5 = self.enc_4(x4)

        x = self.dec_1(x5, x4)
        x = self.dec_2(x, x3)
        x = self.dec_3(x, x2)
        x = self.dec_4(x, x1)

        return self.out_conv(x)
