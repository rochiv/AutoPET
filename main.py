import os
import time
import pandas as pd

from dataset import SegmentationDataset, generate_image_df

from sklearn.model_selection import train_test_split

import torchvision
import torchio as tio
from torch.utils.data import DataLoader
import unet
import torch.optim as optim
from monai.losses import DiceLoss, GeneralizedDiceLoss, MaskedDiceLoss

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

# generate train-test split
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42)

# develop transform
resize_transform = torchvision.transforms.Compose([tio.transforms.Resize(64), torchvision.transforms.Normalize(0.5, 0.5)])

# create train set
train_set = SegmentationDataset(df=train_df, root_dir=data_dir_path, transform=resize_transform)
test_set = SegmentationDataset(df=test_df, root_dir=data_dir_path, transform=resize_transform)

# create data loaders
train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

print(f"Total batches in train_loader: {len(train_loader)}")
print(f"Total batches in test_loader: {len(test_loader)}")

# ---------------------------------------------------------------------------------------------------------

device = 'cuda'

model = unet.UNet3D(in_channels=1, out_classes=1, dimensions=3, padding=1)
model = model.to(device)

learning_rate = 0.001

loss_function = DiceLoss(sigmoid=True)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 20

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    epoch_start_time = time.time()

    for batch_idx, sample in enumerate(train_loader):
        ct, ctres, pet, seg, suv = sample['CT'].to(device), sample['CTres'].to(device), sample['PET'].to(device), \
            sample['SEG'].to(device), sample['SUV'].to(device)

        batch_start_time = time.time()
        print('regular', pet.shape)
        print('float', pet.float().shape)
        # forward pass
        optimizer.zero_grad()
        outputs = model(pet.float())
        print('out', outputs.shape)
        loss = loss_function(outputs, seg)

        loss.backward()
        optimizer.step()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        print(f"  Batch [{batch_idx + 1}/{len(train_loader)}] - Loss: {loss:.4f} - Batch Time: {batch_time:.2f} seconds")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    print(f"Epoch [{epoch + 1}/{num_epochs}] - Total Time: {epoch_time:.2f} seconds")
