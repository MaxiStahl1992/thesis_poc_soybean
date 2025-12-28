import os
from PIL import Image
from torch.utils.data import Dataset

class SoybeanDataset(Dataset):
    """
    Custom Dataset for Soybean Leaf Images.
    Expects folder structure: data_root/dataset_name/Healthy, Rust, Frogeye
    """
    def __init__(self, data_root, dataset_name, transform=None):
        self.data_root = data_root
        self.dataset_name = dataset_name
        self.transform = transform
        self.samples = []
        self.class_to_idx = {'Healthy': 0, 'Rust': 1, 'Frogeye': 2}
        
        if dataset_name.upper() == "ASDID":
            # ASDID: healthy, soyabean_rust, frogeye
            folder_map = {
                'Healthy': 'healthy',
                'Rust': 'soybean_rust',
                'Frogeye': 'frogeye'
            }
        elif dataset_name.upper() == "MH":
            # MH: Healthy_Soyabean, Soyabean_Frog_Leaf_Eye, Soyabean_Rust
            folder_map = {
                'Healthy': 'Healthy_Soyabean',
                'Rust': 'Soyabean_Rust',
                'Frogeye': 'Soyabean_Frog_Leaf_Eye'
            }
        else:
            # Fallback
            folder_map = {name: name for name in self.class_to_idx}
        
        for class_name, class_idx in self.class_to_idx.items():
            folder_name = folder_map.get(class_name, class_name)
            full_path = os.path.join(self.data_root, folder_name)
            
            if os.path.isdir(full_path):
                img_count = 0
                for img_name in os.listdir(full_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(full_path, img_name), class_idx))
                        img_count += 1
                if img_count == 0:
                    print(f"Warning: No images found in {full_path}")
            else:
                print(f"Warning: Could not find folder {full_path} for class {class_name}")

        print(f"Loaded {len(self.samples)} samples for {dataset_name} dataset.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
