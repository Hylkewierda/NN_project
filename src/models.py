"""All CNN architectures for food recognition experiments."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import NUM_CLASSES


# ═══════════════════════════════════════════════════════════════════════
# 1. Baseline CNN — simple 4-layer conv net
# ═══════════════════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    """Simple CNN: 4 conv layers → global avg pool → FC."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════
# 2. Deeper CNN + BatchNorm
# ═══════════════════════════════════════════════════════════════════════

def conv_bn_relu(in_ch, out_ch, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class DeepCNN(nn.Module):
    """Deeper CNN with BatchNorm: 6 conv layers in 3 blocks."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.block1 = nn.Sequential(
            conv_bn_relu(3, 32), conv_bn_relu(32, 32), nn.MaxPool2d(2),
        )
        self.block2 = nn.Sequential(
            conv_bn_relu(32, 64), conv_bn_relu(64, 64), nn.MaxPool2d(2),
        )
        self.block3 = nn.Sequential(
            conv_bn_relu(64, 128), conv_bn_relu(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════
# 3. ResNet-style with skip connections
# ═══════════════════════════════════════════════════════════════════════

class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class FoodResNet(nn.Module):
    """ResNet-style network for food classification.

    Architecture inspired by ResNet-18 but scaled down for training from scratch
    on a smaller dataset: fewer filters, fewer blocks.
    """

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(32, 64, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [ResidualBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_features(self, x):
        """Extract features before classifier (for t-SNE visualization)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# ═══════════════════════════════════════════════════════════════════════
# 4. CNN + Attention (SE-Net / CBAM)
# ═══════════════════════════════════════════════════════════════════════

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)

    def forward(self, x):
        b, c, _, _ = x.size()
        w = x.mean(dim=(2, 3))            # squeeze: global avg pool
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))     # excitation
        return x * w.view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial attention module (part of CBAM)."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.amax(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))
        return x * attn


class CBAMBlock(nn.Module):
    """CBAM = Channel (SE) + Spatial attention after conv block."""

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.se(x)
        x = self.sa(x)
        return x


class AttentionResBlock(nn.Module):
    """Residual block with CBAM attention."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.cbam = CBAMBlock(out_ch)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += self.shortcut(x)
        return F.relu(out, inplace=True)


class AttentionResNet(nn.Module):
    """ResNet + CBAM attention for fine-grained food recognition."""

    def __init__(self, num_classes=NUM_CLASSES, dropout=0.5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(32, 32, 2, stride=1)
        self.layer2 = self._make_layer(32, 64, 2, stride=2)
        self.layer3 = self._make_layer(64, 128, 2, stride=2)
        self.layer4 = self._make_layer(128, 256, 2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _make_layer(self, in_ch, out_ch, num_blocks, stride):
        layers = [AttentionResBlock(in_ch, out_ch, stride)]
        for _ in range(1, num_blocks):
            layers.append(AttentionResBlock(out_ch, out_ch, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def get_features(self, x):
        """Extract features before classifier (for t-SNE visualization)."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


# ═══════════════════════════════════════════════════════════════════════
# Model registry
# ═══════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    "baseline": BaselineCNN,
    "deep_cnn": DeepCNN,
    "resnet": FoodResNet,
    "attention_resnet": AttentionResNet,
}


def get_model(name, **kwargs):
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Choose from {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
