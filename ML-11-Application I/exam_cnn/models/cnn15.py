"""CNN15 architecture and training defaults. Layer names match saved checkpoints."""

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_SIZE = 28
IN_CHANNELS = 1
NUM_CLASSES = 26  # A-Z

# — conv block channel widths (VGG-style: 3x3 convs, filters double
# every time spatial size is halved).
BLOCK_CHANNELS = [32, 64, 128, 256]  # one entry per block, 3 conv layers each = 12 conv layers
CONVS_PER_BLOCK = 3
KERNEL_SIZE = 3
CONV_PADDING = 1  # 'same' padding for a 3x3 kernel

# — dropout schedule, one value per pooling stage + each dense layer
# (block1, block2, block3, dense1, dense2) — block4 has no dropout (feeds GAP).
DROPOUT_SCHEDULE = [0.25, 0.30, 0.35, 0.50, 0.50]

# — classification head (Dense layers 13-15 of the 15).
DENSE_UNITS = [512, 256]  # two hidden dense layers before the final NUM_CLASSES one

# — compile settings.
OPTIMIZER = "adam"  # "adam" | "adamw"
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0  # only used if OPTIMIZER == "adamw"
LABEL_SMOOTHING = 0.05
TOP_K = 3  # secondary metric: top-k accuracy

# Training defaults
BATCH_SIZE = 128
EPOCHS = 100
EARLY_STOPPING_PATIENCE = 12


class ConvBlock(nn.Module):
    """Stack convolution, batch normalization, and ReLU layers."""

    def __init__(self, in_channels: int, out_channels: int, num_convs: int):
        super().__init__()
        layers = []
        for i in range(num_convs):
            c_in = in_channels if i == 0 else out_channels
            layers += [
                nn.Conv2d(c_in, out_channels, KERNEL_SIZE, padding=CONV_PADDING),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
        self.convs = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convs(x)


class CNN15(nn.Module):
    """Twelve convolutional layers and three linear layers."""

    def __init__(self):
        super().__init__()
        channels = [IN_CHANNELS] + BLOCK_CHANNELS

        # BLOCK 1-3: conv block -> maxpool -> dropout
        self.block1 = ConvBlock(channels[0], channels[1], CONVS_PER_BLOCK)
        self.pool1 = nn.MaxPool2d(2)
        self.drop1 = nn.Dropout(DROPOUT_SCHEDULE[0])

        self.block2 = ConvBlock(channels[1], channels[2], CONVS_PER_BLOCK)
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(DROPOUT_SCHEDULE[1])

        self.block3 = ConvBlock(channels[2], channels[3], CONVS_PER_BLOCK)
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(DROPOUT_SCHEDULE[2])

        # BLOCK 4: conv block -> GlobalAveragePooling2D (no maxpool/dropout, per STEP 3.2)
        self.block4 = ConvBlock(channels[3], channels[4], CONVS_PER_BLOCK)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # HEAD: Dense(512) -> BN -> ReLU -> Dropout ; Dense(256) -> ReLU -> Dropout ; Dense(num_classes)
        self.fc1 = nn.Linear(channels[4], DENSE_UNITS[0])
        self.bn_fc1 = nn.BatchNorm1d(DENSE_UNITS[0])
        self.drop_fc1 = nn.Dropout(DROPOUT_SCHEDULE[3])

        self.fc2 = nn.Linear(DENSE_UNITS[0], DENSE_UNITS[1])
        self.drop_fc2 = nn.Dropout(DROPOUT_SCHEDULE[4])

        self.fc3 = nn.Linear(DENSE_UNITS[1], NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop1(self.pool1(self.block1(x)))
        x = self.drop2(self.pool2(self.block2(x)))
        x = self.drop3(self.pool3(self.block3(x)))
        x = self.block4(x)
        x = self.gap(x).flatten(1)  # (N, 256, 1, 1) -> (N, 256)

        x = self.drop_fc1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.drop_fc2(F.relu(self.fc2(x)))
        return self.fc3(x)  # logits, shape (N, NUM_CLASSES)

    def count_weighted_layers(self) -> int:
        """Count convolutional and linear layers."""
        return sum(1 for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear)))


def build_loss() -> nn.Module:
    """Build cross-entropy loss with label smoothing."""
    return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """Build the configured Adam or AdamW optimizer."""
    if OPTIMIZER == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = TOP_K) -> float:
    """Measure how often the correct label appears in the top k predictions."""
    topk = logits.topk(k, dim=1).indices
    correct = topk.eq(targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()


if __name__ == "__main__":
    model = CNN15()
    n_params = sum(p.numel() for p in model.parameters())
    n_weighted_layers = model.count_weighted_layers()

    print(f"Weighted layers (Conv2d + Linear): {n_weighted_layers} (expect 15: 12 conv + 3 dense)")
    print(f"Total parameters: {n_params:,} (plan estimate: ~2.23M)")

    dummy = torch.randn(4, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    out = model(dummy)
    print(f"Forward pass: input {tuple(dummy.shape)} -> output {tuple(out.shape)}")
    assert out.shape == (4, NUM_CLASSES), "output shape mismatch"

    loss_fn = build_loss()
    dummy_targets = torch.randint(0, NUM_CLASSES, (4,))
    loss = loss_fn(out, dummy_targets)
    print(f"Sample loss: {loss.item():.4f}")
    print(f"Sample top-{TOP_K} accuracy: {topk_accuracy(out, dummy_targets):.2f}")
