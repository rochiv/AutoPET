import os
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


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
            image_path = os.path.join(self.root_dir, value)
            image_data = sitk.GetArrayFromImage(sitk.ReadImage(image_path))
            image_tensor = torch.tensor(image_data).unsqueeze(0)  # adding channel

            # apply transform
            if self.transform:
                images_dict[key] = self.transform(image_tensor)
            else:
                images_dict[key] = image_tensor

        return images_dict


def generate_image_df(data_dir: str) -> pd.DataFrame:
    df = pd.DataFrame(columns=['CT', 'CTres', 'PET', 'SUV', 'SEG'])

    patients = os.listdir(data_dir)

    for patient in patients:
        patient_path = os.path.join(data_dir, patient)
        studies = os.listdir(patient_path)

        for study in studies:
            study_path = os.path.join(patient_path, study)
            files = os.listdir(study_path)

            file_path_dict = {'CT': [], 'CTres': [], 'PET': [], 'SUV': [], 'SEG': []}

            for file in files:
                file_rstrip = file.rstrip(".nii.gz")

                if file_rstrip in file_path_dict:
                    file_path_dict[file_rstrip].append(os.path.join(patient, study, file))

            temp_df = pd.DataFrame(file_path_dict)
            df = pd.concat([df, temp_df], ignore_index=True)

    return df
