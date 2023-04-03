import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torchvision.transforms as transforms


class ProstateMRDataset(torch.utils.data.Dataset):
    """Dataset containing prostate MR images. 

    Parameters
    ----------
    paths : list[Path]
        paths to the patient data
    img_size : list[int]
        size of images to be interpolated to
    """
    def __init__(self, paths, img_size):
        self.mr_image_list = []
        self.mask_list = []
        # load images
        for path in paths:
            self.mr_image_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path + "/mr_bffe.mhd"))
            )
            #self.mask_list.append(
            #    sitk.GetArrayFromImage(sitk.ReadImage(path + "/prostaat.mhd"))
            #)
            self.mask_list.append(
                sitk.GetArrayFromImage(sitk.ReadImage(path + "/mr_bffe.mhd"))
            )

        # number of patients and slices in the dataset
        self.no_patients = len(self.mr_image_list)
        self.no_slices = self.mr_image_list[0].shape[0]

        # transforms to resize images
        self.img_transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.CenterCrop(256),
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
        )
        # standardise intensities based on mean and std deviation
        self.train_data_mean = np.mean(self.mr_image_list)
        self.train_data_std = np.std(self.mr_image_list)
        self.norm_transform = transforms.Normalize(
            self.train_data_mean, self.train_data_std
        )

    def __len__(self):
        """Returns length of dataset
        """
        return self.no_patients * self.no_slices


    def __getitem__(self, index):
        """ Returns the preprocessing MR image and corresponding segementation
#         for a given index.
# 
#         Parameters
#         ----------
#         index : int
#             index of the image/segmentation in dataset
#         """

        # compute which slice an index corresponds to
        patient = index // self.no_slices
        the_slice = index - (patient * self.no_slices)

        return (
            self.norm_transform(
                self.img_transform(
                    self.mr_image_list[patient][the_slice, ...].astype(np.float32)
                )
            ),
            self.img_transform(
                (self.mask_list[patient][the_slice, ...] > 0).astype(np.int32)
            ),
        )
 

