import torch
import torch.nn as nn
from torchvision import models

def get_model(num_classes: int, freeze_backbone: bool = True) -> nn.Module:

    model = models.mobilenet_v2(
        weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
    )

    # 🔥 freeze backbone
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # 🔥 replace classifier
    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )

    return model


if __name__ == "__main__":
    NUM_CLASSES = 38  # change later dynamically

    model = get_model(NUM_CLASSES)

    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)

    print("Output shape:", output.shape)
    print("Total params:", sum(p.numel() for p in model.parameters()))
    print("Trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))