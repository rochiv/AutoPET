import os
import pandas as pd

from dataset import SegmentationDataset, generate_image_df

from sklearn.model_selection import train_test_split

import torchvision
import torchio as tio
from torch.utils.data import DataLoader
import unet
import torch.nn as nn
import torch.optim as optim
from monai.losses import DiceLoss

# Enter path to .csv file
csv_path = "image_df.csv"

# Enter path to image data directory
data_dir_path = "/home/rohit/Downloads/nifti/FDG-PET-CT-Lesions"

# Generate image_df.csv if it does not exist
if os.path.exists(csv_path):
    print(f".csv path exists, reading {csv_path}.")
    image_df = pd.read_csv(csv_path)
else:
    print(f".csv path does not exist, generating dataframe and creating {csv_path} file.")
    image_df = generate_image_df(data_dir_path)
    image_df.to_csv(csv_path, index=False)

print(image_df.head)

# generate train-test split
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42)

# develop transform
resize_transform = torchvision.transforms.Compose([tio.transforms.Resize(128)])

# create train set
train_set = SegmentationDataset(df=train_df, root_dir=data_dir_path, transform=resize_transform)
test_set = SegmentationDataset(df=test_df, root_dir=data_dir_path, transform=resize_transform)

# create data loaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

print(f"Total batches in train_loader: {len(train_loader)}")
print(f"Total batches in test_loader: {len(test_loader)}")

for sample in train_loader:
    break

for key, value in sample.items():
    print(f"Key: {key} Value: {value.shape}")

# ---------------------------------------------------------------------------------------------------------

device = 'cuda'

model = unet.UNet3D(in_channels=1, out_classes=2, dimensions=3, padding=1)
model = model.to(device)

learning_rate = 0.001
loss = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 1

for epoch in range(num_epochs):
    for sample in train_loader:
        ct, ctres, pet, seg, suv = sample['CT'].to(device), sample['CTres'].to(device), sample['PET'].to(device), \
        sample['SEG'].to(device), sample['SUV'].to(device)

        # forward pass
        outputs = model(pet.float())
        loss = DiceLoss(outputs, seg)
        optimizer.zero_grad()
        optimizer.step()

print(f"\n\nRESULT LOSS: {loss}")
