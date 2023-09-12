import SimpleITK as sitk
import os
import napari
import numpy as np
import torchio as tio
import napari
import pandas as pd

image_df = pd.read_csv("image_df.csv")
data_dir_path = "/home/rohit/Downloads/nifti/FDG-PET-CT-Lesions"


def sample_to_dict(idx: int) -> (dict, bool):
    sample_subject_path = image_df.iloc[idx]
    sample_subject_dict = sample_subject_path.to_dict()
    sample_subject = {}

    for key, value in sample_subject_dict.items():
        sample_subject[key] = sitk.GetArrayFromImage(sitk.ReadImage(os.path.join(data_dir_path, value)))

    lesion_flag = False
    if np.max(sample_subject["SEG"]) == 1:
        lesion_flag = True

    return sample_subject, lesion_flag


def view_sample(sample_subject: dict) -> None:
    viewer = napari.Viewer()
    viewer.add_image(sample_subject["PET"], name="PET")
    viewer.add_image(sample_subject["SUV"], name="SUV")
    viewer.add_image(sample_subject["CTres"], name="CTres")
    viewer.add_labels(sample_subject["SEG"], name="SEG")
