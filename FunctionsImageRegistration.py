# Script with all functions for image registration part

import elastix 
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import os
import SimpleITK as sitk
import numpy as np
import shutil

def loadPatientDataReadable(patient_nr, filepath):
    """ 
    The MR images and masks are loaded for the specified patient in a 3D array.
    
    input:
    patient_nr: str - choose which patient data to load (e.g. 'p102', 'p107', 'p108', etc.)
    filepath: str - path leading to where your data is stored
    
    output:
    loaded_mr: 3D array - MR images loaded
    loaded_mask: 3D array - masks loaded
    
    """
    file_p_mr = filepath+ '\{}\mr_bffe.mhd'.format(patient_nr)  
    file_p_mask = filepath+'\{}\prostaat.mhd'.format(patient_nr) 

    # Convert so images can be inspected
    loaded_mr = imageio.imread(file_p_mr)
    loaded_mask = imageio.imread(file_p_mask)
    
    return loaded_mr, loaded_mask


def mutualInformation(fixed_patient, moving_patient_slice, slice_nr_fixed):
    """ 
    Calculate mutual information between fixed image slice and moving image slice
    
    input: 
    fixed_patient: 3D array - MR 3D array of patient being used as fixed image
    moving_patient_slice: 2D array - MR 2D array of moving image patient, specific slice 
    slice_nr_fixed: int - slice number of fixed image
    
    output:
    mutual information value
    
    """
    hgram, __, __ = np.histogram2d(fixed_patient[slice_nr_fixed,:,:].ravel(), moving_patient_slice.ravel(), bins=30)
    
    # Convert bins counts to probability values
    pxy = hgram / float(np.sum(hgram))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


def getTwentyNearest(slice_nr_fixed, moving_image_mr, fixed_image_mr):
    """
    Obtain most relevant slices for one patient (moving image) based on given fixed image slice.
    Compute and return the MI between selected slices of moving image and the one selected fixed image slice.
    
    input:
    slice_nr_fixed: int - slice number of fixed image
    moving_image_mr: 3D array - MR 3D array of moving image
    fixed_image_mr: 3D array - MR 3D array of fixed image
    
    output:
    MI_calc: list - MI values in order of slice number (low to high)
    list_index_processed: list - slice numbers that have been used
    
    """
    
    # Create list of indexes (with nothing out of range)
    list_of_index = []
    for j in range(slice_nr_fixed-9, slice_nr_fixed+10):
        list_of_index.append(j)    
    list_index_processed = [item for item in list_of_index if item >= 0 and item < 86] # removes items <0 and >85

    # Perform MI computation part
    MI_calc=[]
    for slices_from_list in list_index_processed:
        MI_calc.append(mutualInformation(fixed_image_mr, moving_image_mr[slices_from_list,:,:], slice_nr_fixed))
    
    return MI_calc, list_index_processed


def allMImovingImages(list_moving, slice_nr_fixed, fixed_image_selected):
    """
    """
    all_MI_per_moving_image = []
    for every_moving_image in list_moving:
        MI_nearest, moving_slices_used = getTwentyNearest(slice_nr_fixed, every_moving_image, fixed_image_selected)
        all_MI_per_moving_image.append(MI_nearest)
        
    return all_MI_per_moving_image, moving_slices_used


def intoOneList(list_of_lists):
    """
    Return lists containing sublists as just one big list.
    """
    big_list = []
    for sublist in list_of_lists:
        for item in sublist:
            big_list.append(item)

    return big_list


def getPatientAndSlice(index_most_similar, used_moving_slices, list_moving, patient_nrs_list_train, list_ids_moving):
    """
    Function to retrieve which patient and which slice index corresponds to the most similar slices.
    """
    divider_patient = len(used_moving_slices) # often 19, not always
    patient_index_grabber = index_most_similar//divider_patient #uses floor function
    slice_grabber = index_most_similar%divider_patient # per patient, which slice index
    actual_slice_nr = used_moving_slices[slice_grabber] # link back to range of slices you are considering
    
    #print('Patient:', list_ids_moving[patient_index_grabber], 'and slice nr:', actual_slice_nr)
    
    return patient_index_grabber, actual_slice_nr


def sort_index(lst, rev=True):
    """ 
    Sort a list from highest to lowest order and return indexes of this new order
    """
    index = range(len(lst))
    s = sorted(index, reverse=rev, key=lambda i: lst[i])
    return s

def visualize_patients(patient_fixed, patient_moving, slice_fixed, slice_moving):
    """
    Visually inspect the slices of the fixed and moving patient
    
    input:
    patient_fixed: patient number of fixed patient (e.g. p107)
    patient_moving: patient number of fixed patient (e.g. p108)
    slice_fixed: slice number of fixed slice
    slice_moving: slice number of moving slice
    
    output:
    plots of fixed and moving slice
    """

    fixed_image_path = filepath_data+ '\{}\mr_bffe.mhd'.format(patient_fixed)
    moving_image_path = filepath_data+ '\{}\mr_bffe.mhd'.format(patient_moving)

    readable_fixed = imageio.imread(fixed_image_path)
    readable_moving = imageio.imread(moving_image_path)

    # visualize the images
    plt.imshow(readable_fixed[slice_fixed,:,:], cmap='gray') 
    plt.title('Fixed MR input image, \nslice {}, {}'.format(slice_fixed, patient_fixed))
    plt.show()

    # instead of manually adjusting this range, this should be the range of the moving slices
    plt.imshow(readable_moving[slice_moving,:,:], cmap='gray') 
    plt.title('Moving MR input image, \nslice {}, {}'.format(slice_moving, patient_moving))

        
def create2DImages(fixed_image_path, fixed_image_ID, filepath_data_stored):
    """
    Create a 2D image of each slice in the 3D image.
    
    input: 
    fixed_image_path: the path to the fixed image slice. 
    fixed_image_ID: ID of patient
    
    output:
    - 
    
    """
    itk_image_fixed = sitk.ReadImage(fixed_image_path)
    #itk_image_moving = sitk.ReadImage(moving_image_path)
    
    # I kept the for-loop since one patient always contains 86 slices and it is so fast that I don't think it is necessary
    # to only create images for certain slices
    for i in range(86): 
        # fixed image
        itk_img_array = sitk.GetArrayFromImage(itk_image_fixed)
        slice_i = sitk.GetImageFromArray(itk_img_array[i,:,:])
        path_fixed = filepath_data_stored+ '\{}\{}_slice{}.mhd'.format(fixed_image_ID, fixed_image_ID, i)
        sitk.WriteImage(slice_i,path_fixed)
    
        ## commented out the moving image, since we now do it for all patients at once
        # moving image
        #itk_img_array_moving = sitk.GetArrayFromImage(itk_image_moving)
        #slice_i = sitk.GetImageFromArray(itk_img_array_moving[i,:,:])
        #path_moving = filepath_data_stored+ '\{}\{}_slice{}.mhd'.format(moving_image_ID, moving_image_ID, i)
        #sitk.WriteImage(slice_i,path_moving)

        
def bspline_registration(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data_stored, el_path):
    """
    Performs bspline transformation
    
    input:
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of moving patient (e.g. p108)
    - slice_fixed: slice number of fixed image
    - slice_moving: slice number of moving imaging
    - filepath_data_stord: str - path leading to data and parameter file
    
    output:
    -
    """
    
    parameter_file_bsplines = os.path.join(filepath_data_stored,'parameters_penalty.txt')
    path_fixed = filepath_data_stored+ '\{}\{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)
    path_moving = filepath_data_stored+ '\{}\{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    
    # create a folder to store the results files
    if not os.path.exists(os.path.join(filepath_data_stored,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,
                                                                                                           slice_fixed,patient_moving,slice_moving))):
        os.makedirs(os.path.join(os.path.join(filepath_data_stored,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,
                                                                                                                          slice_fixed,patient_moving,slice_moving))))
    output_file_path_bspline = os.path.join(os.path.join(filepath_data_stored,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,
                                                                                                                                 slice_fixed,patient_moving,slice_moving)))
    

    # apply the bspline transformation
    el_path.register(
    fixed_image=path_fixed,
    moving_image=path_moving,  
    parameters=[parameter_file_bsplines],
    output_dir=output_file_path_bspline)
    
    
def visualize_bspline_results(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data):
    """
    Visualize results of bspline transformation
    
    input:
     - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - filepath_data: filepath in which all training data is stored
    
    output:
    3 images displayed the fixed slice, moving slice and transformed moving slice
    """
    
    # currently it says slice_fixed, but this should be changed accordingly when it is combined with MI script
    # define the path to the slice of the fixed image
    path_fixed = filepath_data+ '\{}\{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)

    # define the path to the slice of the moving image
    path_moving = filepath_data+ '\{}\{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    
    # create a folder to store the results files
    output_file_path_bspline = os.path.join(os.path.join(filepath_data,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    
    # visualize the images  
    itk_image_fixed = sitk.ReadImage(path_fixed)
    image_array_fixed = sitk.GetArrayFromImage(itk_image_fixed)
    
    itk_image_moving = sitk.ReadImage(path_moving)
    image_array_moving = sitk.GetArrayFromImage(itk_image_moving)
    
    result_path_bspline = os.path.join(output_file_path_bspline, 'result.0.mhd')   
    transformed_moving_image = imageio.imread(result_path_bspline)  
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(image_array_fixed[:,:], cmap='gray')
    ax[1].imshow(image_array_moving[:,:], cmap='gray')
    ax[2].imshow(transformed_moving_image[:,:], cmap='gray')
    
    
def Jacobian(output_file_path_bspline, tr_path):
    """
    Calculate the Jacobian.
    
    input:
    output_file_path_bspline: the path to the TransformParameters.0.txt file.
    tr_path:path to transformix interface
    
    output:
    imb: the Jacobian value.
    
    """
    # define the path to the transform parameters file
    transform_path_im = os.path.join(output_file_path_bspline,'TransformParameters.0.txt')
    
    # apply transformix
    tr = elastix.TransformixInterface(parameters=transform_path_im,
                                  transformix_path=tr_path)
    
    # define the path to the output folder
    output_dir_jacobian = output_file_path_bspline
    jacobian_matrix_path = tr.jacobian_matrix(output_dir=output_dir_jacobian)
    
    # Get the Jacobian determinant
    jacobian_determinant_path = tr.jacobian_determinant(output_dir=output_dir_jacobian)
    
    # Get the full deformation field
    deformation_field_path = tr.deformation_field(output_dir=output_dir_jacobian)
    imb = imageio.imread(jacobian_determinant_path,level=0)
    
    return imb


def visualize_jacobian(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data, tr_path):
    """
    Visualizes jacobian determinant of transformation
    
    input:
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - filepath: path to folder in which all training data is stored
    - tr_path: path to transformix interface
    
    output:
    - image of jacobian determinant in transformed image
    """
    
    output_file_path_bspline = os.path.join(os.path.join(filepath_data,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    
    # calculate the Jacobian per slice
    imb_value = Jacobian(output_file_path_bspline, tr_path=tr_path)
    
    # visualize the Jacobian per slice
    plt.figure(figsize=(5,5))
    pos = plt.imshow(imb_value[:,:],cmap='gray')
    plt.colorbar(pos)
    
def create2DMasks(fixed_mask_path,fixed_image_ID, filepath_data):
    """
    Create a 2D mask of each slice in the 3D mask.
    
    input: 
    fixed_mask_path: the path to the fixed mask slice. 
    fixed_image_ID: ID of patient
    filepath_data: path to training data
    
    output:
    - 
    
    """
    itk_image_fixed = sitk.ReadImage(fixed_mask_path)
    #itk_image_moving = sitk.ReadImage(moving_mask_path)
    
    # I kept the for-loop since one patient always contains 86 slices and it is so fast that I don't think it is necessary
    # to only create masks for certain slices
    for i in range(86): 
        # fixed mask
        itk_img_array = sitk.GetArrayFromImage(itk_image_fixed)
        slice = sitk.GetImageFromArray(itk_img_array[i,:,:])
        path_fixed = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(fixed_image_ID, fixed_image_ID, i)
        sitk.WriteImage(slice,path_fixed)
    
        ## commented this out since we now do it for all patients at once
        # moving image
        #itk_img_array_moving = sitk.GetArrayFromImage(itk_image_moving)
        #slice = sitk.GetImageFromArray(itk_img_array_moving[i,:,:])
        #path_moving = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(fixed_image_ID, moving_image_ID, i)
        #sitk.WriteImage(slice,path_moving)

        
def newTransformParameterFile(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data):
    """
    Create a copy of the 'TransformParameters.0.txt' and rename it to 'TransformParameters.0b.txt'.
    In the new file, the (FinalBSplineInterpolationOrder 3) is changed to value 0.
    
    input:
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - filepath_data: path to training data
    
    output:
    - 
    
    """
    output_file_path_bspline = os.path.join(os.path.join(filepath_data,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    # define the path to the transform parameters file
    transform_path = os.path.join(output_file_path_bspline,'TransformParameters.0.txt')
    
    # create a copy of the file and rename it
    target_path = os.path.join(output_file_path_bspline,'TransformParameters.0b.txt')
    shutil.copyfile(transform_path, target_path)
    
    # change the value from 3 to 0
    with open(target_path,'r') as file:
        data=file.read()
        data=data.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
        
    with open(target_path,'w') as file:
        
        file.write(data)
        
def bspline_mask_registration(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data, tr_path):
    """
    Performs bspline transformation on masks, using the parameter file from the earlier registration
    
    input: 
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - tr_path: path to transformix interface
    - filepath_data: path to folder in which training data is stored
    
    output:
    -
    """

    # define the path to the slice of the fixed mask
    path_fixed_mask = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)

    # define the path to the slice of the moving mask
    path_moving_mask = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    
    # define the path to the transform parameters file
    output_file_path_bspline = os.path.join(os.path.join(filepath_data,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    target_path = os.path.join(output_file_path_bspline,'TransformParameters.0b.txt')
    
    # create a folder to store the results files    
    if not os.path.exists(os.path.join(filepath_data,'{}\\mask_bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving))):
        os.makedirs(os.path.join(os.path.join(filepath_data,'{}\\mask_bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving))))
    output_file_path_bspline_mask = os.path.join(os.path.join(filepath_data,'{}\\mask_bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    
    # apply transformix
    tr = elastix.TransformixInterface(parameters=target_path,
                                  transformix_path=tr_path)
    
    # transform the moving mask with the transformation parameters
    tr.transform_image(path_moving_mask, output_dir=output_file_path_bspline_mask)
    
    
def visualize_bspline_mask_results(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data):
    """
    Visualizes results from bspline registration on the masks:
    
    input:
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - filepath_data: path to training data
    
    output:
    - Fixed image mask, moving image mask and transformed moving image mask
    """


    # define the path to the slice of the fixed mask
    path_fixed_mask = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)

    # define the path to the slice of the moving mask
    path_moving_mask = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    
    
    output_file_path_bspline_mask = os.path.join(os.path.join(filepath_data,'{}\\mask_bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))
    

    # visualize the images  
    itk_image_fixed = sitk.ReadImage(path_fixed_mask)
    image_array_fixed = sitk.GetArrayFromImage(itk_image_fixed)
    
    itk_image_moving = sitk.ReadImage(path_moving_mask)
    image_array_moving = sitk.GetArrayFromImage(itk_image_moving)
    
    result_path_bspline_mask = os.path.join(output_file_path_bspline_mask, 'result.mhd')
    transformed_moving_image = imageio.imread(result_path_bspline_mask)
    
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    ax[0].imshow(image_array_fixed[:,:], cmap='gray')
    ax[1].imshow(image_array_moving[:,:], cmap='gray')
    ax[2].imshow(transformed_moving_image[:,:], cmap='gray')
    
def overlay_mask(patient_fixed, patient_moving, slice_fixed, slice_moving, filepath_data):
    """
    Visualizes registration by overlaying the mask over the MR images
    
    input:
    - patient_fixed: patient number of fixed patient (e.g. p107)
    - patient_moving: patient number of fixed patient (e.g. p108)
    - slice_fixed: slice number of fixed slice
    - slice_moving: slice number of moving slice
    - filepath_data: path to training data
    
    output:
    - Fixed, moving and transformed moving slice with masks overlayed
    """
    
    # define the path to the slice of the fixed image and mask
    path_fixed_IM = filepath_data+ '\{}\{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)
    path_fixed_MASK = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_fixed, patient_fixed, slice_fixed)

    # define the path to the slice of the moving image
    path_moving_IM = filepath_data+ '\{}\{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    output_file_path_IM = os.path.join(os.path.join(filepath_data,'{}\\bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving)))   
    result_path_IM = os.path.join(output_file_path_IM, 'result.0.mhd')
    
    # define the path to the slice of the moving mask
    path_moving_MASK = filepath_data+ '\{}\mask_{}_slice{}.mhd'.format(patient_moving, patient_moving, slice_moving)
    output_file_path_MASK = os.path.join(filepath_data,'{}\\mask_bspline_results_slice_{}\\moving_slice_{}_{}'.format(patient_fixed,slice_fixed,patient_moving,slice_moving))
    result_path_MASK = os.path.join(output_file_path_MASK, 'result.mhd')
    
    # read the images
    readable_fixed_image_path_IM = imageio.imread(path_fixed_IM)
    readable_moving_image_path_IM = imageio.imread(path_moving_IM)
    transformed_moving_image_IM = imageio.imread(result_path_IM)
    
    # read the masks
    itk_image_fixed = sitk.ReadImage(path_fixed_MASK)
    image_array_fixed = sitk.GetArrayFromImage(itk_image_fixed)
    
    itk_image_moving = sitk.ReadImage(path_moving_MASK)
    image_array_moving = sitk.GetArrayFromImage(itk_image_moving)
    
    transformed_moving_image_MASK = imageio.imread(result_path_MASK)
    
    # visualize the images
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    
    ax[0].imshow(readable_fixed_image_path_IM[:,:], cmap='gray')
    ax[0].imshow(image_array_fixed[:,:], cmap='gray', alpha=0.5)
    ax[0].set_title('Fixed image, \nslice {}, {}'.format(slice_fixed, patient_fixed))
    
    ax[1].imshow(readable_moving_image_path_IM[:,:], cmap='gray')
    ax[1].imshow(image_array_moving[:,:], cmap='gray', alpha=0.5)
    ax[1].set_title('Moving image, \nslice {}, {}'.format(slice_moving, patient_moving))
    
    ax[2].imshow(transformed_moving_image_IM[:,:], cmap='gray')
    ax[2].imshow(transformed_moving_image_MASK[:,:], cmap='gray', alpha=0.5)
    ax[2].set_title('Transformed\nmoving image, \nslice {}, {}'.format(slice_moving, patient_moving))
    
    plt.show()
    
def diceFunction(im1, im2):
    '''
    Compute the dice score between two input images or volumes. Note that we use a smoothing factor of 1.
    :param im1: Image 1
    :param im2: Image 2
    :return: Dice score
    '''
    
    readable_im1 = imageio.imread(im1)
    readable_im2 = imageio.imread(im2)
    
    im1 = np.asarray(readable_im1).astype(bool)
    im2 = np.asarray(readable_im2).astype(bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return (2. * intersection.sum() + 1 ) / (im1.sum() + im2.sum() + 1)

def computeQualityMeasures(im_pred, im_truth):
    """
    Computes multiple image registration evaluation measures, including dice score and Hausdorff distances.
    We can decide ourselves which ones we want.
    
    input:
    -im_pred: The predicted segmentation
    -im_truth: The ground truth
    
    output:
    dictionary containing:
    -average Hausdorff distance
    -maximum Hausdorff distance
    -95% Hausdorff distance
    (-dice score)
    (-volume similarity)
    (-false negative rate)
    (-false positive rate)
    (-mean surface distance)
    (-median surface distance)
    (-standard deviation surface distance)
    (-maximum surface distance)
    """
    readable_pred = imageio.imread(im_pred)
    readable_truth = imageio.imread(im_truth)
    
    quality=dict()
    labelPred=sitk.GetImageFromArray(readable_pred, isVector=False)
    labelTrue=sitk.GetImageFromArray(readable_truth, isVector=False)
    
    #Hausdorff Distance
    hausdorffcomputer=sitk.HausdorffDistanceImageFilter()
    hausdorffcomputer.Execute(labelTrue>0.5,labelPred>0.5)
    quality["avgHausdorff"]=hausdorffcomputer.GetAverageHausdorffDistance()
    quality["Hausdorff"]=hausdorffcomputer.GetHausdorffDistance()

    
    ##Dice,Jaccard,Volume Similarity..
    #dicecomputer.Execute(labelTrue>0.5,labelPred>0.5)
    #quality["dice"]=dicecomputer.GetDiceCoefficient()
    #quality["volume_similarity"]=dicecomputer.GetVolumeSimilarity()
    #quality["false_negative"]=dicecomputer.GetFalseNegativeError()
    #quality["false_positive"]=dicecomputer.GetFalsePositiveError()
    
    #Surface distance measures
    label = 1
    ref_distance_map=sitk.Abs(sitk.SignedMaurerDistanceMap(labelTrue>0.5,squaredDistance=False))
    ref_surface=sitk.LabelContour(labelTrue>0.5)
    statistics_image_filter=sitk.StatisticsImageFilter()
    statistics_image_filter.Execute(labelTrue>0.5)
    num_ref_surface_pixels=int(statistics_image_filter.GetSum())

    seg_distance_map=sitk.Abs(sitk.SignedMaurerDistanceMap(labelPred>0.5,squaredDistance=False))
    seg_surface=sitk.LabelContour(labelPred>0.5)
    seg2ref_distance_map=ref_distance_map*sitk.Cast(seg_surface,sitk.sitkFloat32)
    ref2seg_distance_map=seg_distance_map*sitk.Cast(ref_surface,sitk.sitkFloat32)
    
    statistics_image_filter.Execute(labelPred>0.5)
    num_seg_surface_pixels=int(statistics_image_filter.GetSum())

    seg2ref_distance_map_arr=sitk.GetArrayViewFromImage(seg2ref_distance_map)
    seg2ref_distances=list(seg2ref_distance_map_arr[seg2ref_distance_map_arr!=0])
    seg2ref_distances=seg2ref_distances+list(np.zeros(num_seg_surface_pixels-len(seg2ref_distances)))
    ref2seg_distance_map_arr=sitk.GetArrayViewFromImage(ref2seg_distance_map)
    ref2seg_distances=list(ref2seg_distance_map_arr[ref2seg_distance_map_arr!=0])
    ref2seg_distances=ref2seg_distances+list(np.zeros(num_ref_surface_pixels-len(ref2seg_distances)))

    all_surface_distances=seg2ref_distances+ref2seg_distances
    #quality["mean_surface_distance"]=np.mean(all_surface_distances)
    #quality["median_surface_distance"]=np.median(all_surface_distances)
    #quality["std_surface_distance"]=np.std(all_surface_distances)
    #quality["max_surface_distance"]=np.max(all_surface_distances)
    
    
    ### Find the distances to surface points of the contour.  Calculate in both directions         
    dist_seg = sitk.GetArrayViewFromImage(seg_distance_map)[sitk.GetArrayViewFromImage(ref_surface)==1]
    dist_ref = sitk.GetArrayViewFromImage(ref_distance_map)[sitk.GetArrayViewFromImage(seg_surface)==1]


    ### Find the 95% Distance for each direction and average        
    quality['hausdorff_95']= (np.percentile(dist_ref, 95) + np.percentile(dist_seg, 95)) / 2.0

    return quality

def Atlas( seg_data,slices=[]):
    """
    Combine multiple segmentations into one
    
    input:
        - seg_data: list containing readable segmentations
        
    output:
        - STAPLE_seg: combined segmentation
    """
#    segs=['seg_0','seg_1','seg_2','seg_3','seg_4']
#    seg_data=[]
#    for i in range(len(patient_list)):
#        file_p_mask = data_paths+'\{}\prostaat.mhd'.format(patient_list[i]) #Directory will have to be changed still 
#        print(file_p_mask)
#        readable_mask=imageio.imread(file_p_mask)
#        readable_mask=readable_mask[slices[i],:,:]
#        segs[i]= readable_mask
#        seg_data.append(segs[i])
#  
    
    segs_itk=['seg0_itk','seg1_itk','seg2_itk','seg3_itk','seg4_itk']

    seg_stack=[]
    for i in range(len(seg_data)):
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
    