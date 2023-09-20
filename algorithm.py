import os
import time
import numpy as np
import pandas as pd
import torch
from monai.inferers import sliding_window_inference
from tqdm import tqdm
from dataset import SegmentationDataset, generate_image_df
import nibabel as nib

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader
import unet
import torch.optim as optim
from torch.nn import BCEWithLogitsLoss

from val_script import compute_metrics, dice_score
from transforms import resize_normalize, OverlapCrop

torch.autograd.set_detect_anomaly(True)


def generate_image_path_df(csv_path: str, data_dir_path: str) -> pd.DataFrame:
    # Generate image_df.csv if it does not exist
    if os.path.exists(csv_path):
        print(f".csv path exists, reading {csv_path}.")
        image_df = pd.read_csv(csv_path)
    else:
        print(f".csv path does not exist, generating dataframe and creating {csv_path} file.")
        image_df = generate_image_df(data_dir_path)
        image_df.to_csv(csv_path, index=False)
    return image_df


def train(device: str, num_epochs: int, train_loader, test_loader, model, optimizer, loss):
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch + 1}/{num_epochs}]")

        enu_tqdm = tqdm(enumerate(train_loader))
        for batch_idx, sample in enu_tqdm:
            pet, ctres, suv, seg = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)

            pet_ctres_suv = torch.cat((pet, ctres, suv), 1)

            batch_start_time = time.time()

            # forward pass
            optimizer.zero_grad()
            outputs = model(pet_ctres_suv.float())

            bce_loss = loss(outputs, seg)
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
                pet, ctres, suv, seg = sample[0].to(device), sample[1].to(device), sample[2].to(device), sample[3].to(device)
                pet_ctres_suv = torch.cat((pet, ctres, suv), 1)

                # Forward pass
                outputs = model(pet_ctres_suv.float())

                # outputs = torch.where(outputs > 0.5, 1., 0.)

                # Calculate loss
                dice_sc = dice_score(seg, outputs)
                val_loss += dice_sc.item()

                # Update progress bar
                tqdm.write(f"Val Batch [{batch_idx + 1}/{len(test_loader)}] - Dice Score: {dice_sc:.4f}")

        avg_val_loss = val_loss / val_batches
        print(f"Avg Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), f="model6.pt")


def main():
    # path to .csv file
    csv_path = "image_df.csv"

    # path to image data directory
    data_dir_path = "/home/rohit/Downloads/nifti/FDG-PET-CT-Lesions"

    image_df = generate_image_path_df(csv_path, data_dir_path)

    # train-test split
    train_df, test_df = train_test_split(image_df[:64], test_size=0.2, random_state=42)

    # train and test set
    train_set = SegmentationDataset(df=train_df,
                                    root_dir=data_dir_path,
                                    transform_pet=resize_normalize(size=64, component='image'),
                                    transform_suv=resize_normalize(size=64, component='image'),
                                    transform_ctres=resize_normalize(size=64, component='image'),
                                    transform_seg=resize_normalize(size=64, component='mask')
                                    )

    test_set = SegmentationDataset(df=test_df,
                                   root_dir=data_dir_path,
                                   transform_pet=resize_normalize(size=64, component='image'),
                                   transform_suv=resize_normalize(size=64, component='image'),
                                   transform_ctres=resize_normalize(size=64, component='image'),
                                   transform_seg=resize_normalize(size=64, component='mask')
                                   )

    # train and test loaders
    train_loader = DataLoader(train_set, batch_size=4, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=4)

    print(f"Total batches in train_loader: {len(train_loader)}")
    print(f"Total batches in test_loader: {len(test_loader)}")

    device = 'cuda'

    # residual 3D UNet model
    u_net3d = unet.UNet3D(in_channels=3, out_classes=1, dimensions=3, residual=False, padding=1)
    model = u_net3d.to(device)

    # loss function
    bce_loss_function = BCEWithLogitsLoss()

    # learning rate and optimizer
    learning_rate = 0.001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(
        device='cuda',
        num_epochs=10,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        optimizer=optimizer,
        loss=bce_loss_function
    )


def segment_input(ckpt_path: str, data_dir: str, export_dir: str):
    print("starting")

    model = unet.UNet3D(in_channels=3, out_classes=1, dimensions=3, residual=False, padding=1)
    net = model.load_from_checkpoint(ckpt_path)
    net.eval()

    device = torch.device("cuda:0")
    net.to(device)
    net.prepare_data(data_dir)

    with torch.no_grad():
        for i, val_data in enumerate(net.val_dataloader()):
            roi_size = (160, 160, 160)
            sw_batch_size = 4

            mask_out = sliding_window_inference(val_data["image_petct"].to(device), roi_size, sw_batch_size, net)
            mask_out = torch.argmax(mask_out, dim=1).detach().cpu().numpy().squeeze()
            mask_out = mask_out.astype(np.uint8)
            print("done inference")

            PT = nib.load(os.path.join(data_dir, "SUV.nii.gz"))  # needs to be loaded to recover nifti header and export mask
            pet_affine = PT.affine
            PT = PT.get_fdata()
            mask_export = nib.Nifti1Image(mask_out, pet_affine)
            print(os.path.join(export_dir, "PRED.nii.gz"))

            nib.save(mask_export, os.path.join(export_dir, "PRED.nii.gz"))
            print("done writing")


def run_inference(ckpt_path="model5.pt", data_dir='/opt/algorithm/', export_dir="/output/"):  # TODO: add my ckpt
    segment_input(ckpt_path, data_dir, export_dir)


# if __name__ == "__main__":
#     main()
