import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from .soybean import SoybeanDataset

def get_dataloaders(dataset_name, data_root, batch_size=32, train_val_test_split=(0.7, 0.15, 0.15), seed=73):
    """
    Creates DataLoaders for train, val, and test splits.
    Returns (train_loader, val_loader, test_loader, dataset, train_indices, val_indices, test_indices)
    """
    # Standard ConNeXt/Swin/ResNet transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset = SoybeanDataset(data_root, dataset_name, transform=transform)
    
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(train_val_test_split[0] * total_size)
    val_size = int(train_val_test_split[1] * total_size)
    test_size = total_size - train_size - val_size
    
    # Deterministic split
    generator = torch.Generator().manual_seed(seed)
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size], generator=generator)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, dataset, train_ds.indices, val_ds.indices, test_ds.indices
