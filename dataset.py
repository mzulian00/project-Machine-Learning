import torch.utils.data as dutils
import os
from torchvision.io import read_image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from constants import IMG_SIZE, BATCH_SIZE

# Assicurati che la directory esista
dataset_dir = 'dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)
    print(f"Created directory {dataset_dir}. Please add images to this directory.")
else:
    print(f"Directory {dataset_dir} already exists.")

class CustomDataset(dutils.Dataset):
    def __init__(self, transform=None):
        self.image_names = os.listdir(dataset_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = read_image(os.path.join(dataset_dir, image_name)).float() / 255.0  # Normalizzare l'immagine
        if self.transform:
            image = self.transform(image)
        return image

transform = T.Compose([
    T.ToPILImage(),
    T.RandomRotation(10),
    T.CenterCrop(192),
    T.RandomCrop(180),
    T.Resize(IMG_SIZE),
    T.RandomHorizontalFlip(),
    T.ToTensor()
])

dataset = CustomDataset(transform=transform)
dataloader = dutils.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
test_ds = dutils.DataLoader(dataset, batch_size=16, shuffle=True)

if __name__ == '__main__':
    images = next(iter(dataloader))
    plt.figure(figsize=(8, 8))
    plt.title('Images')
    plt.axis('off')
    for i in range(min(len(images), 16)):
        plt.subplot(4, 4, i+1)
        plt.imshow(T.ToPILImage()(images[i]))
    plt.show()