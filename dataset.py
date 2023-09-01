import os

import matplotlib.pyplot as plt
import pandas as pd

import nibabel as nib
import PIL

import torch
import torchio as tio
import torchvision
from torch.utils.data import Dataset, DataLoader


class SegmentationDataset(Dataset):

    def __init__(self, df, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_df = df
        self.transform = transform

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        images_dict = dict(row)

        # convert .nii.gz to tensor
        for key, value in images_dict.items():
            image = nib.load(os.path.join(self.root_dir, value))
            image_data = image.get_fdata()
            image_tensor = torch.tensor(image_data).unsqueeze(0)        # adding channel

            # apply transform
            if self.transform:
                images_dict[key] = self.transform(image_tensor)
            else:
                images_dict[key] = image_tensor

        return images_dict


def display_2d_image_mask(image, mask) -> None:
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    ax1.set_title("Image")
    ax1.imshow(image)

    ax2.set_title("Mask")
    ax2.imshow(mask)


def generate_image_df(data_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['CT', 'CTres', 'PET', 'SUV', 'SEG'])

    patients = os.listdir(data_dir)
    # print(f"patients length = {patients}")

    for patient in patients:
        patient_path = os.path.join(data_dir, patient)
        studies = os.listdir(patient_path)
        # print(f"studies length = {studies}")

        for study in studies:
            study_path = os.path.join(patient_path, study)
            files = os.listdir(study_path)
            # print(f"files length = {files}")

            file_path_dict = {'CT': [], 'CTres': [], 'PET': [], 'SUV': [], 'SEG': []}

            for file in files:
                file_rstrip = file.rstrip(".nii.gz")

                if file_rstrip in file_path_dict:
                    file_path_dict[file_rstrip].append(os.path.join(patient, study, file))

            temp_df = pd.DataFrame(file_path_dict)
            df = pd.concat([df, temp_df], ignore_index=True)

    return df
