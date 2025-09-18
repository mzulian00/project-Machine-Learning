import torch
from dataset import dataloader_test
import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from constants import *
import os
import argparse
from metrics import FID
from metrics import Inception_Score
from metrics import PSNR


def test(args):


	os.chdir('..')
	model = torch.load(os.path.join('drive', [s for s in os.listdir('drive') if s.startswith('My')][0],args.name, 'generator.pkl') , map_location=torch.device('cpu'))
	os.chdir('project-Machine-Learning')
	

	real_images = next(iter(dataloader_test))
	real_images = torch.reshape(real_images, (BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE))

	x_offset = random.randint(0, PATCH_SIZE)
	y_offset = random.randint(0, PATCH_SIZE)

	generated_images = real_images.clone()
	generated_images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

	os.chdir('..')
	os.chdir('drive')
	os.chdir([s for s in os.listdir() if s.startswith('My')][0])
	os.chdir(args.name)
	if os.path.exists('Test') == False:
		os.mkdir('Test')
	os.chdir('Test')

	if args.verbose:
		plt.figure(figsize=(8, 8))
		for i in range(16):
			plt.subplot(4, 4, i+1)
			plt.imshow(T.ToPILImage()(generated_images[i]))
		plt.savefig('Test Empty Patch')

	noise = torch.randn(generated_images.shape[0], 128)
	with torch.no_grad():
		predicted_patches = model(generated_images, noise)
	generated_images[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches

	if args.verbose:
		plt.figure(figsize=(8, 8))
		for i in range(16):
			plt.subplot(4, 4, i+1)
			plt.imshow(T.ToPILImage()(generated_images[i]))
		plt.savefig('Test Filled Patch')

	os.chdir('..')
	os.chdir('..')
	os.chdir('..')
	os.chdir('..')
	os.chdir('project-Machine-Learning')

	print('----- TEST -----')
	# Calculate FID
	fid_value = FID(real_images, generated_images)
	print(f"FID: {fid_value:.4}")

	# Calculate Inception Score
	is_mean, is_std = Inception_Score(generated_images)
	print(f"Inception Score: {is_mean:.4} ± {is_std:.2}")

	# Calculate PSNR
	psnr_value = PSNR(real_images, generated_images)
	print(f"PSNR: {psnr_value:.4}")
	print('----------------')

		  


if __name__ == "__main__":        

	parser = argparse.ArgumentParser()
	parser.add_argument("--name", type=str, default='prova')
	parser.add_argument("--verbose", type=bool, default=True)

	args = parser.parse_args()
	
	test(args)