import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import inception_v3, Inception_V3_Weights
import numpy as np
from scipy.linalg import sqrtm
from scipy.stats import entropy
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from constants import *
import random
from dataset import dataloader_test
import os
import matplotlib.pyplot as plt
import torchvision.transforms as T


def FID(real_images, generated_images):
    """    
    Fréchet Inception Distance (FID):
    Purpose: FID measures the similarity between two sets of images: the generated images and real (or reference) images.
    It is a popular metric in the evaluation of generative models such as GANs (Generative Adversarial Networks).

    How it works: FID compares the distribution of features extracted from the Inception network (a pre-trained neural network)
    for the real and generated images. It calculates the Fréchet distance between the two multivariate Gaussians formed by 
    these feature distributions.

    Key Insight: A lower FID score indicates that the distributions of generated images are closer to the distributions of real images, 
    meaning the generated images are more realistic.
    """
    # Load a pre-trained Inception model
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)  # or use Inception_V3_Weights.DEFAULT
    model.fc = nn.Identity()  # Remove the classification layer
    model.eval()  # Set to evaluation mode

    # Preprocessing to make images suitable for the model
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    
    def get_activations(images):
        images = [preprocess(T.ToPILImage()(img)) for img in images]
        images = torch.stack(images)
        with torch.no_grad():
            activations = model(images).detach().cpu().numpy()
        return activations

    # Get activations for real and generated images
    real_activations = get_activations(real_images)
    generated_activations = get_activations(generated_images)

    # Calculate mean and covariance
    mu_real, sigma_real = np.mean(real_activations, axis=0), np.cov(real_activations, rowvar=False)
    mu_gen, sigma_gen = np.mean(generated_activations, axis=0), np.cov(generated_activations, rowvar=False)

    # Calculate FID score
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real.dot(sigma_gen), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid


"""
def Inception_Score(images, splits=10):

    Inception Score (IS):
    Purpose: IS is used to evaluate the diversity and quality of the images generated by a model.

    How it works: IS uses the same Inception network but focuses on the entropy of the predicted 
    labels for generated images. It considers two factors:
    The model should assign high confidence to a specific class for each generated image 
    (i.e., the image is recognizable as a distinct object).
    The model should generate a diverse set of images across different classes.

    Key Insight: A higher IS means the images are both recognizable and diverse. However, 
    IS does not directly compare the generated images to real images.

    # Load a pre-trained Inception model
    model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, transform_input=False)  # or use Inception_V3_Weights.DEFAULT
    model.fc = nn.Identity()  # Remove the classification layer
    model.eval()  # Set to evaluation mode
    preprocess = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Preprocess and stack images
    images = [preprocess(T.ToPILImage()(img)) for img in images]
    images = torch.stack(images)

    # Get predictions from the Inception model
    with torch.no_grad():
        preds = model(images).detach().cpu().numpy()
    
    # Softmax the outputs to obtain class probabilities
    preds = np.exp(preds) / np.sum(np.exp(preds), axis=1, keepdims=True)

    # Split the predictions into chunks
    split_scores = []
    N = len(images)
    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]

        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            scores.append(entropy(part[i], py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)
"""




def PSNR(original_images, generated_images):
    """
    Peak Signal-to-Noise Ratio (PSNR):
    Purpose: PSNR is a traditional metric for measuring the quality of a reconstructed image compared to a reference image, 
    especially in the context of lossy compression and image processing.

    How it works: PSNR compares the pixel-wise differences between the original image and the generated image, quantifying
    the amount of noise introduced by the inpainting process. It is expressed in decibels (dB), and higher values indicate better quality.
    Formula: 10*log10(MAX^2/MSE) 
    where MAX is the maximum possible pixel value (e.g., 255 for 8-bit images), and MSE is the Mean Squared Error between 
    the original and generated images.

    Key Insight: PSNR values correlate with the pixel-level accuracy of the reconstructed image, making it useful for measuring 
    fidelity but less effective for perceptual quality (i.e., how humans perceive the image).
    """
    
    psnr_values = []
    for orig_img, gen_img in zip(original_images, generated_images):
        psnr_values.append( compare_psnr(np.array(orig_img), np.array(gen_img), data_range=255) )
    return np.mean(psnr_values)



def prova_metrics():
    model = torch.load('models/patch_generator.pkl', map_location=torch.device('cpu'))

    real_image = next(iter(dataloader_test))

    # plt.imshow(T.ToPILImage()(real_image[0]))
    # plt.show()

    x_offset = random.randint(0, PATCH_SIZE)
    y_offset = random.randint(0, PATCH_SIZE)

    fake_image = real_image.clone()
    fake_image[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = 0

    # plt.imshow(T.ToPILImage()(real_image[0]))
    # plt.show()


    noise = torch.randn(fake_image.shape[0], 128)
    with torch.no_grad():
        predicted_patches = model(fake_image, noise)
    fake_image[:, :, x_offset:x_offset+PATCH_SIZE, y_offset:y_offset+PATCH_SIZE] = predicted_patches


    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.imshow(T.ToPILImage()(real_image[0]))
    plt.subplot(1, 2, 2)
    plt.imshow(T.ToPILImage()(fake_image[0]))
    plt.show()

    # Suppose you have two lists of images: real_images and generated_images
    real_images = real_image  # List of PIL images
    generated_images = fake_image  # List of PIL images

    # Calculate FID
    fid_value = FID(real_images, generated_images)
    print(f"FID: {fid_value:.4}")

    # Calculate Inception Score
    is_mean, is_std = Inception_Score(generated_images)
    print(f"Inception Score: {is_mean:.4} ± {is_std:.2}")

    # Calculate PSNR
    psnr_value = PSNR(real_images, generated_images)
    print(f"PSNR: {psnr_value:.4}")



if __name__ == "__main__":        
	prova_metrics()