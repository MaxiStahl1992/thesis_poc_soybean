import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class SoybeanDataset(Dataset):
    """
    Custom Dataset for Soybean Leaf Diseases.
    Supports ASDID (USA) and MH-SoyaHealthVision (India).
    """
    LABEL_MAPPING = {
        'ASDID': {
            'healthy': 0,
            'soybean_rust': 1,
            'frogeye': 2
        },
        'MH': {
            'Healthy_Soyabean': 0,
            'Soyabean_Rust': 1,
            'Soyabean_Frog_Leaf_Eye': 2
        }
    }

    def __init__(self, root_dir, dataset_type, transform=None):
        """
        Args:
            root_dir (str): Root directory of the dataset.
            dataset_type (str): 'ASDID' or 'MH'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.samples = []
        
        mapping = self.LABEL_MAPPING[dataset_type]
        
        # Traverse directories and collect samples
        for folder_name, label in mapping.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                print(f"Warning: Folder {folder_path} not found.")
                continue
            
            for filename in os.listdir(folder_path):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.samples.append((os.path.join(folder_path, filename), label))
                    
        print(f"Loaded {len(self.samples)} samples for {dataset_type} dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def get_dataloaders(dataset_name, data_root, batch_size=32, train_val_test_split=(0.7, 0.15, 0.15), seed=42):
    """
    Creates DataLoaders for train, val, and test splits.
    Returns (train_loader, val_loader, test_loader, dataset, train_indices, val_indices, test_indices)
    """
    # Standard ConNeXt transforms
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
