

dicom = True  # True if the conversion from DICOM to NIfTI is needed
visualize = False
mri_mode = '_t2'
list_visslice = [0, 10, 70] # id of slices to visualize

debug = True # debug mode

loop_all = True # per processare tutto il contenuto di una cartella 
process_mask = True
create_dataset = True # creation of the dataset for deep learning


dataset_folder = '../input/isbi-2015-subject01' # cartelle con immagini - tutti soggetti
mainfold_save = '/kaggle/working/'# folder path dove salvare - dove creare sottocartelle etc.
proc_save = '/kaggle/working/processed' # dove salvar immagini processate
registration_save = '/kaggle/working/registered' # percorso generale dove salvare le registrate

if process_mask == True: 
    mask_folder = '../input/isbi2015masks' # folder with masks
    masksave_folder = '/kaggle/working/mask' # folder where to save masks
    
if dicom == True: 
    dicomfiles_path = '../input/dicompatsc1t2' # path with dicom files
    dicomsave_folder = './fromdicom2nifti' # path where to save nifti files converted from dicom
    converted_name = 'conv.nii' # name of the nifti file

# Registration parameters
type_of_transform='QuickRigid' # tipo di registrazione
atlas_path = '../input/mni-httpnistmnimcgillcaicbm152lin/icbm_avg_152_t2_tal_lin.nii'


# Bias field parameters 
shrink_factor = 4 
convergence={'iters': [50, 50, 50, 50], 'tol': 1e-07}
spline_param=200
verbose=False
weight_mask=None


# Creation of the dataset for deep learning
nii_I2save_path = '../input/myimagesisbi' # folder with NIfTI files to save - IMAGES
nii_M2save_path = '../input/mymaskisbi' # folder with NIfTI files to save - MASKS

save_folderimm = './imm' # folder where to store .png images
save_folderms = './ms' # folder where to store .png masks
from_to =[70, 115] # slices to save to png - the other are excluded


import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import dicom2nifti
import SimpleITK as sitk 
import ants



def nii2np(nii_pathIN):
    # Loads NIfTI file with SimpleITK 
    # INPUT: 
        # nii_pathIN = NIfTI file path
    # OUTPUT:
        # np_im = np array
    
    # print("---* Opening nifti image with SimpleITK... *---") 
    nii_im = sitk.ReadImage(nii_pathIN)
    np_sitk = sitk.GetArrayFromImage(nii_im)    
    np_im = np.zeros((np_sitk.shape[1], np_sitk.shape[2], np_sitk.shape[0]))
        
    for i in range(0, np_sitk.shape[0]):
        np_im[:, :, i] = np_sitk[i, :, :]
    
    return np_im


def dcm2nii(dicom_pathIN, nifti_pathOUT, nifti_name):
    # DICOM folder to NIfTI file conversion with dicom2nifti
    # INPUT:
        # dicom_pathIN =  folder with multiple dicom images that need to be converted to a SINGLE nifti file
        # nifti_pathOUT = folder where to store nifti file --> NB path + the name of the .nii file!
        # nifti_name = name of the nifti file
    # OUTPUT: 
        # nifti_pathOUT =  path of the dicom image
    
    # print('---> Conversion with dicom2nifti function') 
    dicom2nifti.settings.disable_validate_slice_increment()
    
    nifti_pathOUT = folder_creation(nifti_pathOUT) 
    save_d2n = os.path.join(nifti_pathOUT, nifti_name)
    
    dicom2nifti.dicom_series_to_nifti(dicom_pathIN, save_d2n, reorient_nifti=True)

    return nifti_pathOUT


def np2antspy(np_slice):
    # NumPy to ANTsPy format
    # INPUT: 
        # np_slice: npo array (npixel, npixel) - ANTsPy _supported_ntypes = {"uint8", "uint32", "float32", "float64"}
    # OUTPUT: 
        # ants_np_: array convertito
    # print('---> Performing convertion from np array to ANTsImage format...')
    arr = np_slice
    arr_conv = arr.astype('float32')  # CONVERSIONE - indica data type 
    
    # NB casting dopo pipeline? vedo

    arr_conv.dtype.name
    ants_np = ants.from_numpy(arr_conv, origin=None, spacing=None, direction=None, has_components=False,
                              is_rgb=False) # converts the np array to an ANTsImage
    # direction = ants_np.direction # get image direction - type: nd array 
    # print(direction)
    
    return ants_np


def ants2np(im_ants):
    # Conversione da ANTsImage a numpy
    # INPUT: im_ants immagine ants
    # OUTPUT: im_np immagine numpy
    # print('---> Conversion from numpy array to ANTsImage...')
    im_np= im_ants.numpy() # converte ANTs in np array
        
    return im_np

    Dataset load and save function


def niftiDS_loader(dataset_folder, im_name, list_visslice=[], visualize=False):
    # Loads NIfTI dataset and converts to NumPy arrays 
    # INPUT
        # dataset_folder = main folder of the NIfTI image dataset 
        # im_name = name of the file (with file extetion)
        # list_visslice = slices to visualize
        # visualize = for visualization (if debug mode)
    # OUTPUT
        # np_image = image converted to 3D np array
        
    # print("---* Loading the NIfTI dataset from folder %s *---" %dataset_folder) 
    nii_path =  os.path.join(dataset_folder, im_name)
    np_image = nii2np(nii_path) # conversion in numpy array
    
    if visualize == True:
        visualize_nii([nii_path], list_visslice) # nifti slices - not converted
    
    return np_image


def folder_creation(save_path, subf_name=''):
        # Creation of folders + subfolders 
        # INPUT: 
            # save_path: path of the folder
            # subf_name: subfolder name
        # OUTPUT: 
            # subf_path final path of the created folder
        
        subf_path = save_path
        if os.path.exists(save_path)==False:
            os.mkdir(save_path) # folder creation
            
        if subf_name != '':
            subf_path = os.path.join(save_path, subf_name) # subfolder creation
            if os.path.exists(subf_path)==False:
                os.mkdir(subf_path)
                
        return subf_path
    
    
def processed_save(im2save, savefolder, im_name):
    # Save NUMPY array as NIfTI with SimpleITK
    # INPUT: 
        # im2save = iimage as numpy array to save
        #        im_name = im name with no extention
        #        savefolder = cartella dove vengono salvate le immagini
    # OUTPUT: 
        # niisave_folder 
    
    # print("---* Saving images... *---") 
    niisave_folder = folder_creation(savefolder, 'nifti')  # complete path where to save
    
    img = np.zeros((im2save.shape[2], im2save.shape[0], im2save.shape[1]))
    for i in range(0, im2save.shape[2]): # cicla su tutte le slice
         img[i, :, :] = im2save[:, :, i]        
    
    im = np.uint16(img)
    img_sitk = sitk.GetImageFromArray(im)
    sitk.WriteImage(img_sitk, os.path.join(niisave_folder, im_name + '.nii'))

    return niisave_folder

    Visualization functions

# ---- funzioni per la visualizzazione

def visualize_np(np_array, which_slice=[0, 1, 5], title=None):
    # APERTURA DI 1 np array E VISUALIZZAZIONE DI PIU' SLICES (= IMM)
    # INPUT:
    #       np_array: numpy array da visualizzare
    #       which_slice: lista con le fette dell'array da visualizzare
    #       title: stringa, titolo del grafico
    # OUTPUT: None
    
    im = np_array
    slices_list = [] 
    for i in range (0, len(which_slice)):
        slice_num = which_slice[i] # the id number of the slice
        slices_list.append(im[:, :, which_slice[i]])
        
    fig, axes = plt.subplots(1, len(slices_list))
    for i, im_i in enumerate(slices_list):
        axes[i].imshow(im_i, cmap="gray")
        axes[i].set_title('Slice number: %d' % which_slice[i])
    plt.suptitle(title)
    plt.show()
    
    return None


def visualize_nii(paths_list, which_slice = [1], title=None):    
    # Visualizza immagini in file nifti
    # INPUT: paths_list: lista con i percorsi dei file da visualizzare
    #       which_slice: lista con il numero delle fette da visualizzare (es: fetta [0, 1, 5])
    #       title: stringa, titolo grafico
    # OUTPUT: none
    
    # APERTURA DI 1 FILE .nii E VISUALIZZAZIONE DI PIU' SLICES (= IMM) DI QUEL FILE
    if len(paths_list) == 1: # se apro 1 file - visualizzo più slice per quel file
        
        # APERTURA CON SIMPLE ITK
        np_im = nii2np(paths_list[0])
        
        slices_list = [] 
        for i in range (0, len(which_slice)):
            slice_num = which_slice[i] # the id number of the slice
            slices_list.append(np_im[:, :, which_slice[i]])
        
        fig, axes = plt.subplots(1, len(slices_list))
        for i, im_i in enumerate(slices_list):
            axes[i].imshow(im_i, cmap="gray")
            axes[i].set_title('Slice number: %d' % which_slice[i])
            
        plt.suptitle(title)
        plt.show()
     
    # APERTURA DI PIU' FILE .nii E VISUALIZZAZIONE DI UNA STESSA SLICES (= IMM) IN OGNI FILE
    # per confrontare un'immagine processata in diversi step
    else: 
        slices_list = []
        for i in range(len(paths_list)):
            # im = nib.load(paths_list[i])  # loads image before the bias field correction
            # np_im = im.get_fdata() # np array
            
            # APERTURA CON SIMPLE ITK
            np_im = nii2np(paths_list[i])
            slice_num = which_slice[0] # the id number of the slice - equal for all files
            slices_list.append(np_im[:, :, slice_num])
        fig, axes = plt.subplots(1, len(slices_list))
        for i, im_i in enumerate(slices_list):
            axes[i].imshow(im_i, cmap="gray")
            axes[i].set_title('Image %s' % paths_list[i])
        plt.suptitle('Slice number %d' % slice_num)
        plt.show()

    return None


def atlas_registration(atlas_path, im2reg_path, list_visslice, type_of_transform='QuickRigid', visualize=False):
    # Performs image registration to an atlas
    # INPUT: 
        # atlas_path: atlas file (.nii) path
        # im2reg_path: .nii image file path to register
        # type_of_transform: type of transformation for the registration
    # OUTPUT: 
        # im_regANTS: immagine registrata formato ants
        # im_regNP: immagine registrata formato np
    
    # print('---> Atlas registration...')
    if visualize == True: 
        visualize_nii([im2reg_path], list_visslice, title='Immagine nifti prima della registrazione') # visualizzazione del file nii

    atlas_ants = ants.image_read(atlas_path, reorient=False) # loads the atlas
    img_ants = ants.image_read(im2reg_path, reorient=False) # loads the image
    # Computation of the transform
    mytx1 = ants.registration(atlas_ants, img_ants, type_of_transform) 
    # Registration with linear interpolation 
    im_regANTS = ants.apply_transforms(fixed=atlas_ants, moving=img_ants, transformlist=mytx1['fwdtransforms']) # imm registrata
    im_regNP = im_regANTS.numpy() # converte in np da ants e salva
    
    if visualize == True: 
        visualize_np(im_regNP, list_visslice, title='Imm. np dopo la registrazione') # visualizzazione dell'array np
        
    return im_regNP, im_regANTS # ants format


def brainEx_antspy(ants_np, iterations=3, kernel_dim_er=3, kernel_dim_dil=3, kernel_dim_finaler=3, cleanup=2):
    # Brain extraction with ANTsPy 
    # INPUT: 
        # ants_np: ants image to skull strip
    # OUTPUT: 
        # im_step1: immagine formato ANTsImage 
        # mask_final: maschera dell'encefalo
        
    # print('---> Performing brain extraction...')
    original_im = ants_np # ANTsImage
    
        # n erosions
    for i in range(0, iterations):
        im_eroded = ants.morphology(ants_np, 'erode', kernel_dim_er, mtype='grayscale', shape='cross') 
        ants_np = im_eroded
    # ants.plot(ants_np)
        # n-1 dilatations
    for i in range(0, iterations-1): 
        im_dilat = ants.morphology(im_eroded, 'dilate', kernel_dim_dil, mtype='grayscale', shape='cross') 
        im_eroded = im_dilat
    # ants.plot(im_eroded)

    mask_opened = ants.get_mask(im_dilat)
    # ants.plot(mask_opened)
    im_eroded_final = ants.morphology(im_dilat, 'erode', kernel_dim_finaler, mtype='grayscale', shape='cross') 
    mask_final = ants.get_mask(im_eroded_final, cleanup=cleanup)
    # ants.plot(mask_final)

    im_step1 = mask_final * original_im # final ANTsImage
    
    return im_step1, mask_final


def biasCorr_antspy(brainex_im, mask_im, shrink_factor=4, convergence={'iters': [50, 50, 50, 50], 'tol': 1e-07}, spline_param=200,
                   verbose=False, weight_mask=None):
    # INPUT brainex_im: brain extracted image
    #       mask_im: the mask obtained from the brain extraction step
    # OUTPUT: bias corrected ANTsImage
    
    # print('---> Performing n4 bias field correction...')
    im_corrected = ants.n4_bias_field_correction(brainex_im, mask=None, shrink_factor=shrink_factor, 
                                                 convergence=convergence, spline_param=spline_param,
                                                verbose=verbose, weight_mask=weight_mask)    
    
    return im_corrected


def sum_masks(mask_folder, subjct=5, timepnt=5, n_masks=2):
    # Sum of 2 lesions masks - done for ISBI dataset 
    # INPUT 
    #     mask_folder: folder con maschere
    #     subj= numero di soggetti
    #     timepnt = numero MASSIMO di time points per oggetto
    # OUTPUT restituisce percorso folder dove ha salvato le maschere
    
    # print('---* Mask processing... *---')
    mask_path = mask_folder # path con le maschere 
    list_masks = os.listdir(mask_path) # lista contenuto 
    
    for i in range(0, subjct): 
        final_mask = np.zeros(shape=(217, 181, 181)) # sti numeri da levare
        for jj in range(0, timepnt):
            base_name = 'training0' + str(i+1) + '_0' + str(jj+1) # nome ID dell'immagine 
            
            for s in range(0, len(list_masks)):
                if base_name in list_masks[s]: # non è detto che ci siano tutti i time points per ogni soggetto
                    finalmask_name = base_name + '_M' # suffisso _M identifica la maschera
                    np_mask = niftiDS_loader(mask_path, list_masks[s], list_visslice, visualize=False) # apro come np array
                    # im_rot = np.rot90(np_mask, k=1, axes=(1,0)) # ruota
                    # im_flip = np.flip(im_rot, 0) # flip ???
                    # np_mask = im_flip
                    
                    for k in range(0, np_mask.shape[2]): # for loop that processes each slice 
                        mask_slice = np_mask[:, :, k] # slice to process                        
                        if n_masks == 2: 
                            unified_masks = np.zeros((np_mask.shape[0], np_mask.shape[1]))
                            for mask in [mask_slice, final_mask[:, :, k]]:
                                unified_masks += mask

                            unified_masks[unified_masks > 1] = 1 # metto tutto a 1
                            final_mask[:, :, k] = unified_masks 
                    processed_save(final_mask, masksave_folder, finalmask_name)
                    
    return masksave_folder

    Creation of the dataset for deep learning applications
    
def network_datasetcreation(nii_path, nii_name, save_folder, saveas='png', from_to=[0, 0], exclude=True, masks=True):
    # Salva file nifti in png - funziona per SINGOLO FILE
    # INPUT: nii_path path del singolo file .NIfTI
    #        nii_name nome del file
    #        save_folder cartella main dove salvare png o jpg (messe in sottocartelle con il nome dell'estensione)
    #        saveas lista con formati in cui salvare 
    #        from_to: includo solo immagini da ... a INCLUSE
    #        exclude = True: non esclude immagini per cui ha fallito pre-proc
    #        masks = True: se salvo le maschere devo avere il nome con _M al fondo!
    # OUTPUT: None
    
    # print("---* Creating the dataset for the neural network...*---") 
    
    np_imm = nii2np(os.path.join(nii_path, nii_name)) # apro nifti con simple itk 
    
    if exclude == True: # non considero alcune slice dell'immagine DI NON INTERESSE PER LA SEGMENTAZIONE 
        np_imm = np_imm[:, :, from_to[0]:from_to[1]]
   
    save_path = folder_creation(save_folder, subf_name=saveas) # creo sottocartella con format

    for i in range(0, np_imm.shape[2]): # salva ogni slice come png
        np_save = np_imm[:, :, i]
        data = np_save / np_save.max() # normalizza intensità per salvataggio
        data = 255 * data # scale by 255
        img = data.astype(np.uint8)
        im = Image.fromarray(img) # immagine da salvare
                
        if saveas=='png':
            if masks == True: # per il nome della maschera con _M al fondo!
                save_name = nii_name.split('_M')[0]  + '_' + str(i) + '_M' + '.png'
            else:
                save_name = nii_name.split('.')[0]  + '_' + str(i) + '.png'
                
            im.save(os.path.join(save_path, save_name)) # sistemare il percorso - nome del file
            
    return None



if dicom == True: #
    dcm2nii(dicomfiles_path, dicomsave_folder, converted_name) # converts from dicom and saves to nifti

    Image registration

# REGISTRATION 
im_path2reg = dataset_folder # immagini da registrare
list_images2reg = os.listdir(im_path2reg) # lista contenuto 

if debug == True: # takes only 1 image file for the debug
    list_images2reg = [list_images2reg[0]]
    print(list_images2reg)

for i in range(0, len(list_images2reg)): 
    if 'T2' in list_images2reg[i].upper(): # only processed t2 images and exclude the others in the folder
        print('--> Registrazione img: ', list_images2reg[i])

        reg_save = folder_creation(registration_save) # folder where to save registered images
        im2reg = os.path.join(im_path2reg, list_images2reg[i])
        im_reg_name = list_images2reg[i].split(mri_mode)[0] # rimuovo estensione dal nome del file e _t2 che indica il formato

        # type_of_transform='QuickRigid' # tipo di registrazione
        im_registeredNP, im_registeredANTS = atlas_registration(atlas_path, im2reg, list_visslice, type_of_transform, visualize=visualize)

        im_rot = np.rot90(im_registeredNP, k=1, axes=(1,0)) # ruota
        im_flip = np.flip(im_rot, 0)

        if visualize == True: 
            visualize_np(im_flip, [0, 10, 100], title='Registered np image slices') # visualizzazione del file nii

        reg_impath = processed_save(im_flip, reg_save, im_reg_name) # saves as nifti
        # print(reg_impath)

# PROCESSING 
list_registeredIM = os.listdir(reg_impath) # lista contenuto

if debug == True:
    list_registeredIM = [list_registeredIM[0]]

for i in range(0, len(list_registeredIM)):
    im_name = list_registeredIM[i]

    np_image = niftiDS_loader(reg_impath, im_name, list_visslice, visualize=visualize)

    #                          -------- * PROCESSING PIPELINE *--------
    # print('----> Looping inside the np array, processing each slice....') 

    np_shape = np_image.shape 
    # print('* The shape of the image slice is: ', np_shape, ' *')
    # print('* There are ', np_shape[2], ' image slices to process *')

    image_processed = np.zeros(shape=(np_shape[0], np_shape[1], np_shape[2])) # initialize final np array
    final_masks = np.zeros(shape=(np_shape[0], np_shape[1], np_shape[2])) # initialize final np array

    for i in range(0, np_image.shape[2]): # for loop that processes each slice 
        # print('---> Processing image nr. ', i)
        im_slice = np_image[:, :, i] # slice to process

        #                     -------- * conversion from np array to ANTsImage * -------- 
        im_ants = np2antspy(im_slice)

        #                          -------- * brain extraction * -------- 
        brain_extracted, mask_extracted = brainEx_antspy(im_ants, iterations=3, 
                                                         kernel_dim_er=3, kernel_dim_dil=3, kernel_dim_finaler=3, cleanup=2)  

        #                             -------- * bias field correction * --------

        im_corr = biasCorr_antspy(brain_extracted, mask_extracted, shrink_factor, convergence, spline_param, verbose, weight_mask) # with ANTsPy

        #                          ------ * final conversion from ANTsImage to np array * -------

        slice_processed = ants2np(im_corr)  # Converte ANTs in np array

        final_mask = ants2np(mask_extracted)  # Converte ANTs in np array
        # 
        image_processed[:, :, i] = slice_processed
        final_masks[:, :, i] = final_mask

    if visualize == True:
        visualize_np(image_processed, list_visslice) # visualizzazione np imm. processate

    #                         --------------- * salvataggio * -------------

    im_name_save = im_name.split('.') # il nome dell'immagine: cosa viene prima dell'estensione 
    final_imname = im_name_save[0] # gestisco il nome dei file
    # print(final_imname)

    processed_save(image_processed, proc_save, final_imname)

# Processing of the masks - sum of the masks of 2 readers (done for ISBI dataset)
if process_mask == True: 
    # FATTO PER DATASET ISBI - 5 soggetti e fino a 5 time points per soggetto
    subjct = 5 # numero di soggetti 
    timepnt = 5 # max di time points per soggetto
    n_masks = 2 # number of masks to sum
    
    sum_masks(mask_folder, subjct, timepnt, n_masks) # somma delle maschere



# CREATION OF THE DATASET FOR DEEP LEARNING

if create_dataset == True: 
    
    # far girare nel folder la funzione
    im_list = os.listdir(nii_I2save_path)

    # salvo in png immagini 
    for i in range(0, len(im_list)):
        network_datasetcreation(nii_I2save_path, im_list[i], save_folderimm, saveas='png', from_to, exclude=True)
    
    if process_mask == True:
        ms_list = os.listdir(nii_M2save_path)
        # salvo in png maschere
        for i in range(0, len(ms_list)):
            network_datasetcreation(nii_M2save_path, ms_list[i], save_folderms, saveas='png', from_to=, exclude=True)

