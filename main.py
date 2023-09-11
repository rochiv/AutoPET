import os
import time
import pandas as pd
import torch

from dataset import SegmentationDataset, generate_image_df

from sklearn.model_selection import train_test_split

import torchvision
import torchio as tio
from torch.utils.data import DataLoader
import unet
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

# generate train-test split
train_df, test_df = train_test_split(image_df, test_size=0.2, random_state=42)

# develop transform
resize_transform = torchvision.transforms.Compose([
    tio.transforms.Resize(100, image_interpolation="linear"),
    tio.transforms.RandomFlip(),
    tio.transforms.RandomBlur(),
    tio.transforms.ZNormalization(),        # FIXME: ZNormalization() transform results in an error
])

# create train set
train_set = SegmentationDataset(df=train_df, root_dir=data_dir_path, transform=resize_transform)
test_set = SegmentationDataset(df=test_df, root_dir=data_dir_path, transform=resize_transform)

# create data loaders
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=2)

print(f"Total batches in train_loader: {len(train_loader)}")
print(f"Total batches in test_loader: {len(test_loader)}")

# ---------------------------------------------------------------------------------------------------------


device = 'cuda'

model = unet.UNet3D(in_channels=2, out_classes=1, dimensions=3, padding=1)
model = model.to(device)
# ddp_model = DDP(model, device_ids=[torch.cuda.device_count()])      # FIXME: default proc group has not been initialized, call init_process_group

learning_rate = 0.001

dice_loss_function = DiceLoss(sigmoid=True)
gen_dice_loss_function = GeneralizedDiceLoss()
masked_dice_loss_function = MaskedDiceLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10

training_start_time = time.time()

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    epoch_start_time = time.time()

    for batch_idx, sample in enumerate(train_loader):
        ct, ctres, pet, seg, suv = sample['CT'].to(device), sample['CTres'].to(device), sample['PET'].to(device), \
            sample['SEG'].to(device), sample['SUV'].to(device)

        pet_suv = torch.cat((pet, suv), 1)

        batch_start_time = time.time()

        # forward pass
        optimizer.zero_grad()
        outputs = model(pet_suv.float())  # TODO: add other data as channels

        dice_loss = dice_loss_function(outputs, seg)

        dice_loss.backward()

        optimizer.step()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        print(f"        Batch [{batch_idx + 1}/{len(train_loader)}] - Dice: {dice_loss:.4f} - Batch Time: {batch_time:.2f} seconds")

    epoch_end_time = time.time()
    epoch_time = epoch_end_time - epoch_start_time

    if epoch % 5 == 0:
        torch.save(model.state_dict(), f=f"model3_epoch{epoch + 1 // 5}.pt")
        print(f"Model saved at epoch #{epoch}")

    print(f"    Epoch [{epoch + 1}/{num_epochs}] - Total Time: {epoch_time:.2f} seconds")

training_end_time = time.time()
training_time = training_end_time - training_start_time

print(f"Training time: {training_time:.2f} seconds")

torch.save(model.state_dict(), f="model2.pt")
