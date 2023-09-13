import os
import time
import pandas as pd
import torch
from tqdm import tqdm
from dataset import SegmentationDataset, generate_image_df

from sklearn.model_selection import train_test_split

import torchvision
import torchio as tio
from torch.utils.data import DataLoader
import unet
import torch.optim as optim
from monai.losses import DiceLoss
from torch.nn import BCEWithLogitsLoss

from val_script import compute_metrics, dice_score

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
resize_transform_pet = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="linear"),
    tio.transforms.ZNormalization(),
])

resize_transform_suv = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="linear"),
    tio.transforms.ZNormalization(),
])

resize_transform_seg = torchvision.transforms.Compose([
    tio.transforms.Resize(64, image_interpolation="nearest"),
])

# create train set
train_set = SegmentationDataset(df=train_df,
                                root_dir=data_dir_path,
                                transform_pet=resize_transform_pet,
                                transform_suv=resize_transform_suv,
                                transform_seg=resize_transform_seg
                                )

test_set = SegmentationDataset(df=test_df,
                               root_dir=data_dir_path,
                               transform_pet=resize_transform_pet,
                               transform_suv=resize_transform_suv,
                               transform_seg=resize_transform_seg
                               )

# create data loaders
train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=4, shuffle=False, num_workers=4)

print(f"Total batches in train_loader: {len(train_loader)}")
print(f"Total batches in test_loader: {len(test_loader)}")

# ----------------------------------------------------------------------------------------------------------------------

# TODO: create main function
# TODO: create separate functions for transform, dataloader

device = 'cuda'

model = unet.UNet3D(in_channels=2, out_classes=1, dimensions=3, padding=1)
model = model.to(device)

learning_rate = 0.001

bce_loss_function = BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
num_epochs = 10

for epoch in range(num_epochs):
    print(f"Epoch [{epoch + 1}/{num_epochs}]")

    enu_tqdm = tqdm(enumerate(train_loader))
    for batch_idx, sample in enu_tqdm:
        pet, suv, seg = sample[0].to(device), sample[1].to(device), sample[2].to(device)

        pet_suv = torch.cat((pet, suv), 1)

        batch_start_time = time.time()

        # forward pass
        optimizer.zero_grad()
        outputs = model(pet_suv.float())

        bce_loss = bce_loss_function(outputs, seg)
        bce_loss.backward()

        optimizer.step()

        enu_tqdm.set_description(f"Batch [{batch_idx + 1}/{len(train_loader)}] - BCE: {bce_loss:.4f}")

    # validation
    model.eval()
    val_loss = 0.0
    val_batches = len(test_loader)

    with torch.no_grad():
        val_tqdm = tqdm(enumerate(test_loader))
        for batch_idx, sample in val_tqdm:
            pet, suv, seg = sample[0].to(device), sample[1].to(device), sample[2].to(device)
            pet_suv = torch.cat((pet, suv), 1)

            # Forward pass
            outputs = model(pet_suv.float())

            outputs = torch.where(outputs > 0.5, 1., 0.)

            # Calculate loss
            dice_sc = dice_score(seg, outputs)
            val_loss += dice_sc.item()

            # Update progress bar
            tqdm.write(f"Val Batch [{batch_idx + 1}/{len(test_loader)}] - Dice Score: {dice_sc:.4f}")

    avg_val_loss = val_loss / val_batches
    print(f"Avg Validation Loss: {avg_val_loss:.4f}")

torch.save(model.state_dict(), f="model4.pt")
