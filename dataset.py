import os
import pandas as pd
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):

    def __init__(self, df, root_dir, transform_pet, transform_suv, transform_seg):
        self.root_dir = root_dir
        self.image_df = df
        self.transform_pet = transform_pet
        self.transform_suv = transform_suv
        self.transform_seg = transform_seg

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, idx):
        row = self.image_df.iloc[idx]
        images_dict = dict(row)

        pet = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, images_dict['PET'])))
        suv = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, images_dict['SUV'])))
        seg = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(self.root_dir, images_dict['SEG'])))

        pet_tensor = torch.tensor(pet).unsqueeze(0)
        suv_tensor = torch.tensor(suv).unsqueeze(0)
        seg_tensor = torch.tensor(seg).unsqueeze(0)

        pet_transform = self.transform_pet(pet_tensor)
        suv_transform = self.transform_suv(suv_tensor)
        seg_transform = self.transform_seg(seg_tensor)

        return pet_transform.float(), suv_transform.float(), seg_transform.float()


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
