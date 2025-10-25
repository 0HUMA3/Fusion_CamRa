# src/models/radardepthnet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RadarDepthNet(nn.Module):
    """
    Simple encoder-fusion-decoder network that fuses image + radar to predict dense depth.
    - image: [B,3,H,W]
    - radar: [B, 3, N]   (or [B, 3*N])
    Output:
    - depth: [B,1,H,W]
    """
    def __init__(self, radar_dim=3, radar_points=128, pretrained_backbone=False):
        super().__init__()
        # Encoder: ResNet18 trunk (remove avgpool & fc)
        backbone = models.resnet18(weights=None)
        if pretrained_backbone:
            try:
                backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            except Exception:
                backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-2])  # -> [B,512,h,w]
        self.enc_ch = 512

        # Radar MLP -> produce spatial feature map compatible with encoder spatial dims
        self.radar_mlp = nn.Sequential(
            nn.Linear(radar_dim * radar_points, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
        )
        self.radar_feat_ch = 128

        # Bottleneck conv to reduce concatenated channels
        self.bottleneck = nn.Conv2d(self.enc_ch + self.radar_feat_ch, 256, kernel_size=3, padding=1)

        # Decoder: series of ConvTranspose2d to upsample approx 32x -> input size
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True)
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True)
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True)
        )
        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.dec5 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16), nn.ReLU(inplace=True)
        )

        # Final conv to single-channel depth
        self.outconv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        # initialization
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, img, radar):
        """
        img: [B,3,H,W]
        radar: either [B,3,N] or [B, 3*N]
        returns: depth [B,1,H,W]
        """
        b = img.shape[0]
        enc = self.encoder(img)   # [B,512,h_enc,w_enc]
        _, _, h_enc, w_enc = enc.shape

        # Prepare radar features
        r = radar
        if r.ndim == 3:
            # [B,3,N] -> flatten
            r = r.view(b, -1)
        elif r.ndim == 2:
            r = r
        else:
            r = r.view(b, -1)
        rfeat = self.radar_mlp(r)                 # [B, radar_feat_ch]
        # to spatial
        rfeat = rfeat.view(b, self.radar_feat_ch, 1, 1).expand(-1, -1, h_enc, w_enc)

        # fuse
        fused = torch.cat([enc, rfeat], dim=1)    # [B, enc_ch + radar_ch, h_enc, w_enc]
        x = self.bottleneck(fused)                # [B,256,h_enc,w_enc]

        # decode
        x = self.dec1(x)   # up x2
        x = self.dec2(x)   # up x2
        x = self.dec3(x)   # up x2
        x = self.dec4(x)   # up x2
        x = self.dec5(x)   # up x2 (should roughly match input H,W)
        out = self.outconv(x)  # [B,1,H_pred,W_pred]

        # final upsample to exactly input resolution if needed
        H_in, W_in = img.shape[2], img.shape[3]
        if out.shape[2] != H_in or out.shape[3] != W_in:
            out = F.interpolate(out, size=(H_in, W_in), mode='bilinear', align_corners=False)
        return out
