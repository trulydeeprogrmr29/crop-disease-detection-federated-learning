from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(data_dir, batch_size=32, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle
    )

    return loader, dataset.classes

