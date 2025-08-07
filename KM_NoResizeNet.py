"""Model without resize or crop based on 8,300-parameter ImprovedNet."""
import torch
import torch.nn as nn
from torchvision import transforms

# Transformation without resize or crop
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

class ImprovedNet(nn.Module):
    """CNN with ~8,300 parameters using adaptive pooling.

    Unlike the original training script, this model expects images to
    be provided at their original size since no resize or crop is
    performed in the preprocessing pipeline.
    """
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 5, padding=2),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = ImprovedNet()
    print(f"Total parameters: {count_parameters(model)}")
    dummy = torch.randn(1, 1, 120, 60)
    out = model(dummy)
    print("Output shape:", out.shape)
