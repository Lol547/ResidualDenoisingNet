import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DevisingDataset(Dataset):
    def __init__(self, root_dir, split='train'):
        self.clean_dir = os.path.join(root_dir, split, 'clean')
        self.noisy_dir = os.path.join(root_dir, split, 'noisy')
        self.filenames = sorted(os.listdir(self.clean_dir))
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        clean_path = os.path.join(self.clean_dir, filename)
        noisy_path = os.path.join(self.noisy_dir, filename)

        clean_img = Image.open(clean_path).convert('RGB')
        noisy_img = Image.open(noisy_path).convert('RGB')

        return self.transform(noisy_img), self.transform(clean_img)
