from dataset import dataloader_test, dataloader_train

images = next(iter(dataloader_test))
print(f"Original shape: {images.shape}")

images = next(iter(dataloader_train))
print(f"Original shape: {images.shape}")

