from dataset import dataloader_train
from generator import Generator
from discriminator import Discriminator
from constants import *
from tqdm import tqdm
from torch import nn
import torch
import random
import torchvision.transforms as T
import torchvision.utils as vutils
import time
import os
import sys

print(f'Started training using device: {device} - {EPOCHS}')

generator = Generator().to(device)
discriminator = Discriminator().to(device)

d_opt = torch.optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))
g_opt = torch.optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2))

loss_fn = nn.BCELoss()

fixed_noise = torch.randn(BATCH_SIZE, Z_DIM, device=device)
fixed_images = next(iter(dataloader_train))[:BATCH_SIZE].to(device)

fixed_x_offset = random.randint(0, PATCH_SIZE)
fixed_y_offset = random.randint(0, PATCH_SIZE)

start = time.time()
for epoch in range(EPOCHS):
	for image_batch in tqdm(dataloader_train):
		image_batch = image_batch.to(device)
		b_size = image_batch.shape[0]
		discriminator.zero_grad()

		y_hat_real = discriminator(image_batch).view(-1)
		y_real = torch.ones_like(y_hat_real, device=device)
		real_loss = loss_fn(y_hat_real, y_real)
		real_loss.backward()

		# Make part of the image black
		x_offset = random.randint(0, PATCH_SIZE)
		y_offset = random.randint(0, PATCH_SIZE)
		image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

		# Predict using generator
		noise = torch.randn(b_size, Z_DIM, device=device)
		predicted_patch = generator(image_batch, noise)

		# Replace black patch with generator output
		image_batch[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patch

		# Predict fake images using discriminator
		y_hat_fake = discriminator(image_batch.detach()).view(-1)

		# Train discriminator
		y_fake = torch.zeros_like(y_hat_fake)
		fake_loss = loss_fn(y_hat_fake, y_fake)
		fake_loss.backward()
		d_opt.step()

		# Train generator
		generator.zero_grad()
		y_hat_fake = discriminator(image_batch).view(-1)
		g_loss = loss_fn(y_hat_fake, torch.ones_like(y_hat_fake))
		g_loss.backward()
		g_opt.step()

	fixed_images[:, :, fixed_x_offset:fixed_x_offset+PATCH_SIZE, fixed_y_offset:fixed_y_offset+PATCH_SIZE] = 0
	with torch.no_grad():
		predicted_patches = generator(fixed_images, fixed_noise)
	fixed_images[:, :, fixed_x_offset:fixed_x_offset+PATCH_SIZE, fixed_y_offset:fixed_y_offset+PATCH_SIZE] = predicted_patches
	img = T.ToPILImage()(vutils.make_grid(fixed_images.to('cpu'), normalize=True, padding=2, nrow=4))
	img.save(os.path.join('progress', f'epoch_{epoch}.jpg'))
train_time = time.time() - start
print(f'Total training time: {train_time // 60} minutes')

generator = generator.to('cpu')

torch.save(generator, os.path.join('models', 'patch_generator.pkl'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("no arguments")
    else:
        print(f"{len(sys.argv)} arguments")