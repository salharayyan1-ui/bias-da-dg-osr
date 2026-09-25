"""Task 4 Step 3: GCSC ("stronger closed-set classifier"). Identical to
Vanilla except RandAugment(num_ops=2, magnitude=9) is inserted after the
crop+flip and before ToTensor/Normalize; everything else (init, optimizer,
schedule, batch size, epochs, seed, checkpoint rule) is unchanged."""
from __future__ import annotations

from task4.methods._common import OSRTrainConfig, train_classifier


def run(model, train_loader, val_loader, logger, cfg: OSRTrainConfig | None = None):
    cfg = cfg or OSRTrainConfig()
    return train_classifier(model, train_loader, val_loader, cfg, logger, run_name="gcsc")


def build_train_transform():
    from torchvision import transforms
    CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
    CIFAR10_STD = (0.2023, 0.1994, 0.2010)
    return transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
