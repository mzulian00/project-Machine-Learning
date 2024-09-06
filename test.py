import torch
from dataset import dataloader_test
import random
import matplotlib.pyplot as plt
import torchvision.transforms as T
from constants import *


model = torch.load('models/generator.pkl', map_location=torch.device('cpu'))

images = next(iter(dataloader_test))
images = torch.reshape(images, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))

x_offset = random.randint(0, PATCH_SIZE)
y_offset = random.randint(0, PATCH_SIZE)

images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

plt.figure(figsize=(8, 8))
for i in range(BATCH_SIZE):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(images[i]))
plt.show()

noise = torch.randn(images.shape[0], 128)
with torch.no_grad():
	predicted_patches = model(images, noise)
images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches

plt.figure(figsize=(8, 8))
for i in range(16):
	plt.subplot(4, 4, i+1)
	plt.imshow(T.ToPILImage()(images[i]))
plt.show()
