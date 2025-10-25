import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class RadarNet(nn.Module):
    def __init__(self, radar_dim=3, radar_points=128, out_classes=10):
        super(RadarNet, self).__init__()

        # 1️⃣ Image Encoder (ResNet backbone)
        backbone = models.resnet18(weights=None)
        self.image_encoder = nn.Sequential(*list(backbone.children())[:-2])  # Remove avgpool, fc
        self.img_feat_dim = 512  # resnet18 last conv feature dim

        # 2️⃣ Radar Encoder (simple MLP)
        self.radar_encoder = nn.Sequential(
            nn.Linear(radar_dim * radar_points, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # 3️⃣ Depth Acc (DACC) Encoder (small CNN)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 4️⃣ Fusion Layer
        self.fusion = nn.Sequential(
            nn.Linear(self.img_feat_dim + 128 + 32, 256),
            nn.ReLU(),
            nn.Linear(256, out_classes)
        )

    def forward(self, image, radar, dacc):
        # Image features
        img_feat = self.image_encoder(image)           # [B, 512, H/32, W/32]
        img_feat = F.adaptive_avg_pool2d(img_feat, 1)  # [B, 512, 1, 1]
        img_feat = img_feat.view(img_feat.size(0), -1) # [B, 512]

        # Radar features
        radar = radar.view(radar.size(0), -1)          # [B, 3*N]
        radar_feat = self.radar_encoder(radar)         # [B, 128]

        # DACC features
        depth_feat = self.depth_encoder(dacc)          # [B, 32, H/4, W/4]
        depth_feat = F.adaptive_avg_pool2d(depth_feat, 1).view(dacc.size(0), -1)  # [B, 32]

        # Fuse all
        fused = torch.cat([img_feat, radar_feat, depth_feat], dim=1)
        out = self.fusion(fused)
        return out
