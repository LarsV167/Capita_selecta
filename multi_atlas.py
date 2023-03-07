import SimpleITK as sitk 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import numpy as np



def Atlas(patient_list=[],data_paths='path',slices=[]):
    segs=['seg_0','seg_1','seg_2','seg_3','seg_4']
    seg_data=[]
    for i in range(len(patient_list)):
        file_p_mask = data_paths+'\{}\prostaat.mhd'.format(patient_list[i]) #Directory will have to be changed still 
        print(file_p_mask)
        readable_mask=imageio.imread(file_p_mask)
        readable_mask=readable_mask[slices[i],:,:]
        segs[i]= readable_mask
        seg_data.append(segs[i])
    
    segs_itk=['seg0_itk','seg1_itk','seg2_itk','seg3_itk','seg4_itk']

    seg_stack=[]
    for i in range(len(segs)):
         segs_itk[i]= sitk.GetImageFromArray(seg_data[i].astype(np.int16))
         seg_stack.append(segs_itk[i])


# Run STAPLE algorithm
    STAPLE_seg_sitk = sitk.STAPLE(seg_stack,1, 1.0)# 1.0 specifies the foreground value


# convert back to numpy array
    STAPLE_seg = sitk.GetArrayFromImage(STAPLE_seg_sitk)

# I made these loops below so that you get a binary mask and not the probabilities of the combinations.
# So all values ​​below 0.5 have a high probability of being zero and anything above that of being 1. Possible discussion point
    for i in range(STAPLE_seg.shape[0]):
        for j in range(STAPLE_seg.shape[1]):
            if STAPLE_seg[i,j]<=0.5:
                STAPLE_seg[i,j]=0
            else:
                STAPLE_seg[i,j]=1
    return STAPLE_seg