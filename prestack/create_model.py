#!/usr/bin/env python
# coding: utf-8

# Importing required libraries to run the code #
import os
from pathlib import Path
import numpy as np
import pandas as pd
import segyio  # to read seismic using Equinor's library
import tensorflow as tf
from empatches import EMPatches
from keras import backend as k
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from keras.models import Model
import json

# input directory where datafiles are stored
input_dir = r"C:\GeoHackaton2023"

# By default, output results to current directory
output_dir = os.getcwd()

# far offset #
# importing input training dataset without AGC
pathlist = Path(input_dir).glob('**/*_PreSTM_final_far.sgy')  # iterate through the folder
all_dataset_seismic_final_far = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r', ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        all_dataset_seismic_final_far.append(data)

# split the dataset to make sure you are training on the appropriate dataset.
seismic_dataset_far = [all_dataset_seismic_final_far[i] for i in [0, 1, 3, 4, 5, 6, 7]]  # for training dataset
unseen_seismic_dataset_far = all_dataset_seismic_final_far[2]  # keeping blind dataset

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
seismic_dataset_far[0] = seismic_dataset_far[0].T[300:1101, 15450:20166]
seismic_dataset_far[1] = seismic_dataset_far[1].T[500:1101, 10475:18801]
seismic_dataset_far[2] = seismic_dataset_far[2].T[500:1101, 10470:18251]
seismic_dataset_far[3] = seismic_dataset_far[3].T[500:1101, 7000:13951]
seismic_dataset_far[4] = seismic_dataset_far[4].T[500:1101, 8475:10951]
seismic_dataset_far[5] = seismic_dataset_far[5].T[500:1101, 2800:6801]
seismic_dataset_far[6] = seismic_dataset_far[6].T[500:1101, 4400:7276]
del all_dataset_seismic_final_far  # free up memory storage

# mid offset #
pathlist = Path(input_dir).glob('**/*_PreSTM_final_mid.sgy')
all_dataset_seismic_final_mid = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r', ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        all_dataset_seismic_final_mid.append(data)

# split the dataset to make sure you are training on the appropriate dataset.
seismic_dataset_mid = [all_dataset_seismic_final_mid[i] for i in [0, 1, 3, 4, 5, 6, 7]]  # for training dataset
unseen_seismic_dataset_mid = all_dataset_seismic_final_mid[2]  # keeping blind dataset

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
seismic_dataset_mid[0] = seismic_dataset_mid[0].T[300:1101, 15450:20166]
seismic_dataset_mid[1] = seismic_dataset_mid[1].T[500:1101, 10475:18801]
seismic_dataset_mid[2] = seismic_dataset_mid[2].T[500:1101, 10470:18251]
seismic_dataset_mid[3] = seismic_dataset_mid[3].T[500:1101, 7000:13951]
seismic_dataset_mid[4] = seismic_dataset_mid[4].T[500:1101, 8475:10951]
seismic_dataset_mid[5] = seismic_dataset_mid[5].T[500:1101, 2800:6801]
seismic_dataset_mid[6] = seismic_dataset_mid[6].T[500:1101, 4400:7276]
del all_dataset_seismic_final_mid  # free up memory storage

# near offset #
pathlist = Path(input_dir).glob('**/*_PreSTM_final_near.sgy')
all_dataset_seismic_final_near = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    # print(path_in_str)
    with segyio.open(path_in_str, 'r', ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        all_dataset_seismic_final_near.append(data)

# split the dataset to make sure you are training on the appropriate dataset.
seismic_dataset_near = [all_dataset_seismic_final_near[i] for i in [0, 1, 3, 4, 5, 6, 7]]  # for training dataset
unseen_seismic_dataset_near = all_dataset_seismic_final_near[2]  # keeping blind dataset   

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
seismic_dataset_near[0] = seismic_dataset_near[0].T[300:1101, 15450:20166]
seismic_dataset_near[1] = seismic_dataset_near[1].T[500:1101, 10475:18801]
seismic_dataset_near[2] = seismic_dataset_near[2].T[500:1101, 10470:18251]
seismic_dataset_near[3] = seismic_dataset_near[3].T[500:1101, 7000:13951]
seismic_dataset_near[4] = seismic_dataset_near[4].T[500:1101, 8475:10951]
seismic_dataset_near[5] = seismic_dataset_near[5].T[500:1101, 2800:6801]
seismic_dataset_near[6] = seismic_dataset_near[6].T[500:1101, 4400:7276]
del all_dataset_seismic_final_near  # free up memory space        

# combine datasets combining near mid far for dataset
seismic_dataset = []
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[0][..., np.newaxis], seismic_dataset_mid[0][..., np.newaxis],
     seismic_dataset_far[0][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[1][..., np.newaxis], seismic_dataset_mid[1][..., np.newaxis],
     seismic_dataset_far[1][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[2][..., np.newaxis], seismic_dataset_mid[2][..., np.newaxis],
     seismic_dataset_far[2][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[3][..., np.newaxis], seismic_dataset_mid[3][..., np.newaxis],
     seismic_dataset_far[3][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[4][..., np.newaxis], seismic_dataset_mid[4][..., np.newaxis],
     seismic_dataset_far[4][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[5][..., np.newaxis], seismic_dataset_mid[5][..., np.newaxis],
     seismic_dataset_far[5][..., np.newaxis]], axis=2))
seismic_dataset.append(np.concatenate(
    [seismic_dataset_near[6][..., np.newaxis], seismic_dataset_mid[6][..., np.newaxis],
     seismic_dataset_far[6][..., np.newaxis]], axis=2))

del seismic_dataset_near  # free up memory
del seismic_dataset_mid
del seismic_dataset_far

# Importing Acoustic Inversion dataset #
pathlist = Path(input_dir).glob('**/*_AcousticImpedance.sgy')
acoustic_impedance_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        acoustic_impedance_dataset.append(data)

# Transpose acoustic impedance dataset for visual comparison for time = y-coordinates
acoustic_impedance_dataset[0] = acoustic_impedance_dataset[0].T
acoustic_impedance_dataset[1] = acoustic_impedance_dataset[1].T
acoustic_impedance_dataset[2] = acoustic_impedance_dataset[2].T
acoustic_impedance_dataset[3] = acoustic_impedance_dataset[3].T
acoustic_impedance_dataset[4] = acoustic_impedance_dataset[4].T
acoustic_impedance_dataset[5] = acoustic_impedance_dataset[5].T
acoustic_impedance_dataset[6] = acoustic_impedance_dataset[6].T

# Importing Bulk Modulus dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_BulkModulus.sgy')
bulk_modulus_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        bulk_modulus_dataset.append(data)

# Transpose dataset
bulk_modulus_dataset[0] = bulk_modulus_dataset[0].T
bulk_modulus_dataset[1] = bulk_modulus_dataset[1].T
bulk_modulus_dataset[2] = bulk_modulus_dataset[2].T
bulk_modulus_dataset[3] = bulk_modulus_dataset[3].T
bulk_modulus_dataset[4] = bulk_modulus_dataset[4].T
bulk_modulus_dataset[5] = bulk_modulus_dataset[5].T
bulk_modulus_dataset[6] = bulk_modulus_dataset[6].T

# Importing Density dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_Density.sgy')
density_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        density_dataset.append(data)

# Transpose dataset
density_dataset[0] = density_dataset[0].T
density_dataset[1] = density_dataset[1].T
density_dataset[2] = density_dataset[2].T
density_dataset[3] = density_dataset[3].T
density_dataset[4] = density_dataset[4].T
density_dataset[5] = density_dataset[5].T
density_dataset[6] = density_dataset[6].T

# Importing Permeability dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_Permeability.sgy')
permeability_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        permeability_dataset.append(data)

# Transpose dataset
permeability_dataset[0] = permeability_dataset[0].T
permeability_dataset[1] = permeability_dataset[1].T
permeability_dataset[2] = permeability_dataset[2].T
permeability_dataset[3] = permeability_dataset[3].T
permeability_dataset[4] = permeability_dataset[4].T
permeability_dataset[5] = permeability_dataset[5].T
permeability_dataset[6] = permeability_dataset[6].T

# Importing Poisson Ratio dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_PoissonsRatio.sgy')
poissonratio_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        poissonratio_dataset.append(data)

# Transpose dataset
poissonratio_dataset[0] = poissonratio_dataset[0].T
poissonratio_dataset[1] = poissonratio_dataset[1].T
poissonratio_dataset[2] = poissonratio_dataset[2].T
poissonratio_dataset[3] = poissonratio_dataset[3].T
poissonratio_dataset[4] = poissonratio_dataset[4].T
poissonratio_dataset[5] = poissonratio_dataset[5].T
poissonratio_dataset[6] = poissonratio_dataset[6].T

# Importing Porosity dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_Porosity.sgy')
porosity_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        porosity_dataset.append(data)

# Transpose dataset
porosity_dataset[0] = porosity_dataset[0].T
porosity_dataset[1] = porosity_dataset[1].T
porosity_dataset[2] = porosity_dataset[2].T
porosity_dataset[3] = porosity_dataset[3].T
porosity_dataset[4] = porosity_dataset[4].T
porosity_dataset[5] = porosity_dataset[5].T
porosity_dataset[6] = porosity_dataset[6].T

# Importing Shear Impedance dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_ShearImpedance.sgy')
shear_impedance_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        shear_impedance_dataset.append(data)

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
shear_impedance_dataset[0] = shear_impedance_dataset[0].T
shear_impedance_dataset[1] = shear_impedance_dataset[1].T
shear_impedance_dataset[2] = shear_impedance_dataset[2].T
shear_impedance_dataset[3] = shear_impedance_dataset[3].T
shear_impedance_dataset[4] = shear_impedance_dataset[4].T
shear_impedance_dataset[5] = shear_impedance_dataset[5].T
shear_impedance_dataset[6] = shear_impedance_dataset[6].T

# Importing Shear Modulus dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_ShearModulus.sgy')
shear_modulus_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        shear_modulus_dataset.append(data)

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
shear_modulus_dataset[0] = shear_modulus_dataset[0].T
shear_modulus_dataset[1] = shear_modulus_dataset[1].T
shear_modulus_dataset[2] = shear_modulus_dataset[2].T
shear_modulus_dataset[3] = shear_modulus_dataset[3].T
shear_modulus_dataset[4] = shear_modulus_dataset[4].T
shear_modulus_dataset[5] = shear_modulus_dataset[5].T
shear_modulus_dataset[6] = shear_modulus_dataset[6].T

# Importing Vp_Vs dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_VpVs.sgy')
Vp_Vs_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        Vp_Vs_dataset.append(data)

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
Vp_Vs_dataset[0] = Vp_Vs_dataset[0].T
Vp_Vs_dataset[1] = Vp_Vs_dataset[1].T
Vp_Vs_dataset[2] = Vp_Vs_dataset[2].T
Vp_Vs_dataset[3] = Vp_Vs_dataset[3].T
Vp_Vs_dataset[4] = Vp_Vs_dataset[4].T
Vp_Vs_dataset[5] = Vp_Vs_dataset[5].T
Vp_Vs_dataset[6] = Vp_Vs_dataset[6].T

# Importing Young Modulus dataset using same method as above #
pathlist = Path(input_dir).glob('**/*_YoungsModulus.sgy')
YoungsModulus_dataset = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r') as segyfile:
        data = segyfile.trace.raw[:]
        YoungsModulus_dataset.append(data)

# clipping the seismic training dataset based on the extents from the inversion
# ranges are based on the pdf document provided and the ranges as displayed in OpendTect divided by two for TWT
YoungsModulus_dataset[0] = YoungsModulus_dataset[0].T
YoungsModulus_dataset[1] = YoungsModulus_dataset[1].T
YoungsModulus_dataset[2] = YoungsModulus_dataset[2].T
YoungsModulus_dataset[3] = YoungsModulus_dataset[3].T
YoungsModulus_dataset[4] = YoungsModulus_dataset[4].T
YoungsModulus_dataset[5] = YoungsModulus_dataset[5].T
YoungsModulus_dataset[6] = YoungsModulus_dataset[6].T

# AUGMENTING DATASET USING PATCHES OF EQUAL SIZE #
# # Augmenting dataset. Creating patches using emp
emp = EMPatches()

patches_seismic = []
for a in seismic_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256, 3])
    patches_seismic.append(img_patches)
patches_seismic = tf.expand_dims(np.concatenate(patches_seismic, axis=0), -1)
print(patches_seismic.shape)

# do the same for acoustic impedance dataset
patches_acoustic_impedance = []
for a in acoustic_impedance_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_acoustic_impedance.append(img_patches)
patches_acoustic_impedance = tf.expand_dims(np.concatenate(patches_acoustic_impedance, axis=0), -1)
print(patches_acoustic_impedance.shape)

# do the same for bulk modulus dataset
patches_bulk_modulus = []
for a in bulk_modulus_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_bulk_modulus.append(img_patches)
patches_bulk_modulus = tf.expand_dims(np.concatenate(patches_bulk_modulus, axis=0), -1)
print(patches_bulk_modulus.shape)

# do the same for density dataset
patches_density = []
for a in density_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_density.append(img_patches)
patches_density = tf.expand_dims(np.concatenate(patches_density, axis=0), -1)
print(patches_density.shape)

# do the same for permeability dataset
patches_permeability = []
for a in permeability_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_permeability.append(img_patches)
patches_permeability = tf.expand_dims(np.concatenate(patches_permeability, axis=0), -1)
print(patches_permeability.shape)

# # do the same for poisson ratio dataset
patches_poissonratio = []
for a in poissonratio_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_poissonratio.append(img_patches)
patches_poissonratio = tf.expand_dims(np.concatenate(patches_poissonratio, axis=0), -1)
print(patches_poissonratio.shape)

# # # do the same for porosity dataset
patches_porosity = []
for a in porosity_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_porosity.append(img_patches)
patches_porosity = tf.expand_dims(np.concatenate(patches_porosity, axis=0), -1)
print(patches_porosity.shape)

# do the same for shear_impedance dataset
patches_shear_impedance = []
for a in shear_impedance_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_shear_impedance.append(img_patches)
patches_shear_impedance = tf.expand_dims(np.concatenate(patches_shear_impedance, axis=0), -1)
print(patches_shear_impedance.shape)

# do the same for shear modulus dataset
patches_shear_modulus = []
for a in shear_modulus_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_shear_modulus.append(img_patches)
patches_shear_modulus = tf.expand_dims(np.concatenate(patches_shear_modulus, axis=0), -1)
print(patches_shear_modulus.shape)

# do the same for Vp Vs dataset
patches_Vp_Vs = []
for a in Vp_Vs_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_Vp_Vs.append(img_patches)
patches_Vp_Vs = tf.expand_dims(np.concatenate(patches_Vp_Vs, axis=0), -1)
print(patches_Vp_Vs.shape)

# do the same for Youngs Modulus dataset
patches_YoungsModulus = []
for a in YoungsModulus_dataset[0:5]:
    img_patches, indices = emp.extract_patches(a, patchsize=256, overlap=0.5)
    img_patches = np.concatenate(img_patches)
    img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
    patches_YoungsModulus.append(img_patches)
patches_YoungsModulus = tf.expand_dims(np.concatenate(patches_YoungsModulus, axis=0), -1)
print(patches_YoungsModulus.shape)

# REMOVE PATCHES OF TOO MANY ZEROS #

# remove the patches with too many zeros
# https://stackoverflow.com/questions/71939370/what-can-be-a-faster-way-to-iterate-through-the-pixels-of-an-image
indices_without_zero = []
for i in range(0, len(tf.squeeze(patches_acoustic_impedance))):
    image = tf.squeeze(patches_acoustic_impedance)[i]
    x, y = np.where(image == 0)  # find the number of pixels with background
    ratio_background = len(x) / (image.shape[1] * image.shape[0])  # get ratio
    if ratio_background < 0.25:
        indices_without_zero.append(i)  # only keep indices that have less background
indices_without_zero = np.array(indices_without_zero)

no_background_seismic = np.asarray(patches_seismic)[indices_without_zero]  # seismic
no_background_ai = np.asarray(patches_acoustic_impedance)[indices_without_zero]  # acoustic impedance
no_background_bulk_modulus = np.asarray(patches_bulk_modulus)[indices_without_zero]  # bulk modulus
no_background_density = np.asarray(patches_density)[indices_without_zero]  # density
no_background_permeability = np.asarray(patches_permeability)[indices_without_zero]  # permeability
no_background_poissonratio = np.asarray(patches_poissonratio)[indices_without_zero]  # poisson ratio
no_background_porosity = np.asarray(patches_porosity)[indices_without_zero]  # porosity
no_background_shear_impedance = np.asarray(patches_shear_impedance)[indices_without_zero]  # shear impedance
no_background_shear_modulus = np.asarray(patches_shear_modulus)[indices_without_zero]  # shear modulus
no_background_Vp_Vs = np.asarray(patches_Vp_Vs)[indices_without_zero]  # VpVs
no_background_Youngs_Modulus = np.asarray(patches_YoungsModulus)[indices_without_zero]  # Youngs Modulus

# FLIP DATASET FOR ADDITIONAL DATA #

# flip seismic
updated = tf.image.flip_left_right(tf.squeeze(no_background_seismic))
resized_seismic = np.concatenate([tf.squeeze(no_background_seismic), updated], axis=0)
print(resized_seismic.shape)

# flip acoustic impedance
updated = tf.image.flip_left_right(no_background_ai)
resized_acoustic_impedance = np.concatenate([no_background_ai, updated], axis=0)

# flip bulk modulus
updated = tf.image.flip_left_right(no_background_bulk_modulus)
resized_bulk_modulus = np.concatenate([no_background_bulk_modulus, updated], axis=0)

# flip density
updated = tf.image.flip_left_right(no_background_density)
resized_density = np.concatenate([no_background_density, updated], axis=0)

# flip permeability
updated = tf.image.flip_left_right(no_background_permeability)
resized_permeability = np.concatenate([no_background_permeability, updated], axis=0)

# # flip porosity
updated = tf.image.flip_left_right(no_background_porosity)
resized_porosity = np.concatenate([no_background_porosity, updated], axis=0)

# flip poisson ratio
updated = tf.image.flip_left_right(no_background_poissonratio)
resized_poissonratio = np.concatenate([no_background_poissonratio, updated], axis=0)

# flip shear impedance
updated = tf.image.flip_left_right(no_background_shear_impedance)
resized_shear_impedance = np.concatenate([no_background_shear_impedance, updated], axis=0)

# flip shear modulus
updated = tf.image.flip_left_right(no_background_shear_modulus)
resized_shear_modulus = np.concatenate([no_background_shear_modulus, updated], axis=0)

# flip Vp Vs
updated = tf.image.flip_left_right(no_background_Vp_Vs)
resized_Vp_Vs = np.concatenate([no_background_Vp_Vs, updated], axis=0)

# flip Young Modulus
updated = tf.image.flip_left_right(no_background_Youngs_Modulus)
resized_Youngs_Modulus = np.concatenate([no_background_Youngs_Modulus, updated], axis=0)

# delete variables to free up memory
del patches_seismic
del indices_without_zero
del patches_acoustic_impedance
del patches_bulk_modulus
del patches_density
del updated
del no_background_seismic
del no_background_ai
del no_background_bulk_modulus
del no_background_density

# SAVE SETTINGS TO A CONFIG FILE TO be loading in the next py file #
# https://stackoverflow.com/questions/19078170/python-how-would-you-save-a-simple-settings-config-file
config = {
    "seismic": [np.mean(resized_seismic, dtype='float64'), np.std(resized_seismic, dtype='float64')],
    "acoustic_impedance": [np.mean(resized_acoustic_impedance, dtype='float64'),
                           np.std(resized_acoustic_impedance, dtype='float64')],
    "bulk_modulus": [np.mean(resized_bulk_modulus, dtype='float64'), np.std(resized_bulk_modulus, dtype='float64')],
    "density": [np.mean(resized_density, dtype='float64'), np.std(resized_density, dtype='float64')],
    "permeability": [np.mean(resized_permeability, dtype='float64'), np.std(resized_permeability, dtype='float64')],
    "poissonratio": [np.mean(resized_poissonratio, dtype='float64'), np.std(resized_poissonratio, dtype='float64')],
    "porosity": [np.mean(resized_porosity, dtype='float64'), np.std(resized_porosity, dtype='float64')],
    "shear_impedance": [np.mean(resized_shear_impedance, dtype='float64'),
                        np.std(resized_shear_impedance, dtype='float64')],
    "shear_modulus": [np.mean(resized_shear_modulus, dtype='float64'), np.std(resized_shear_modulus, dtype='float64')],
    "Vp_Vs": [np.mean(resized_Vp_Vs, dtype='float64'), np.std(resized_Vp_Vs, dtype='float64')],
    "Youngs_Modulus": [np.mean(resized_Youngs_Modulus, dtype='float64'),
                       np.std(resized_Youngs_Modulus, dtype='float64')],
}

with open('config.json', 'w') as f:
    json.dump(config, f)

# NORMALIZE USING STANDARD SCALER #
# scaling factor standardscaler
print('seismic')
scaled_seismic = np.float32(resized_seismic - config["seismic"][0]) / config["seismic"][1]

print('acoustic impedance')
scaled_acousticimpedance = (np.float32(resized_acoustic_impedance - config["acoustic_impedance"][0])
                            / config["acoustic_impedance"][1])

print('bulk modulus')
scaled_bulk_modulus = np.float32(resized_bulk_modulus - config["bulk_modulus"][0]) / config["bulk_modulus"][1]

print('density')
scaled_density = np.float32(resized_density - config["density"][0]) / config["density"][1]

print('permeability')
scaled_permeability = np.float32(resized_permeability - config["permeability"][0]) / config["permeability"][1]

print('poisson ratio')
scaled_poissonratio = np.float32(resized_poissonratio - config["poissonratio"][0]) / config["poissonratio"][1]

print('porosity')
scaled_porosity = np.float32(resized_porosity - config["porosity"][0]) / config["porosity"][1]

print('shear impedance')
scaled_shear_impedance = (np.float32(resized_shear_impedance - np.mean(resized_shear_impedance))
                          / np.std(resized_shear_impedance))

print('shear modulus')
scaled_shear_modulus = (np.float32(resized_shear_modulus - np.mean(resized_shear_modulus))
                        / np.std(resized_shear_modulus))

print('Vp Vs')
scaled_Vp_Vs = np.float32(resized_Vp_Vs - np.mean(resized_Vp_Vs)) / np.std(resized_Vp_Vs)

print('Youngs Modulus')
scaled_Youngs_Modulus = (np.float32(resized_Youngs_Modulus - np.mean(resized_Youngs_Modulus))
                         / np.std(resized_Youngs_Modulus))

# CREATE THE LOSS FUNCTIONS TO BE USED FOR MODEL BUILDING #
# # Model building
# Create an L2 regularizer
l2_regularizer = regularizers.l2(0.01)


def regularized_loss_masked(y_true, y_pred):
    # Calculate the Mean Squared Error
    loss = k.mean(k.square(y_pred * k.cast(y_true > tf.reduce_min(y_true), "float32") - y_true), axis=-1)

    # Add the L2 regularization
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            loss += l2_regularizer(layer.kernel)

    return loss


# Define a custom R-squared metric
def r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


def adjusted_r_squared(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate the sum of squared residuals
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))

    # Calculate the total sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    # Calculate the number of observations and the number of independent variables
    n = tf.shape(y_true)[0]

    # y_pred has a single dimension for the number of predictors
    k = 1

    k = tf.cast(k, tf.float32)
    ss_res = tf.cast(ss_res, tf.float32)
    ss_tot = tf.cast(ss_tot, tf.float32)
    n = tf.cast(n, tf.float32)

    # Calculate adjusted R-squared

    adjusted_r2 = 1 - (ss_res / (ss_tot * (n - 1) / (n - k)))

    return adjusted_r2


# https://stackoverflow.com/questions/41707621/keras-mean-squared-error-loss-layer
# https://stackoverflow.com/questions/57037128/keras-custom-loss-ignore-zero-labels

# CREATE THE MODEL ARCHITECTURE U-NET #
def unet():
    inputs = Input(shape=(None, None, num_channels))

    # Encoder
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Center
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # Decoder
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # Output layer with linear activation for regression
    acousticimpedance = Conv2D(1, (1, 1), activation='linear',
                               name='acoustic_impedance')(conv9)
    bulkmodulus = Conv2D(1, (1, 1), activation='linear',
                         name='bulk_modulus')(conv9)
    density = Conv2D(1, (1, 1), activation='linear',
                     name='density')(conv9)
    # facies = Conv2D(1, (1, 1), activation='linear',
    #                 name='facies')(conv9)
    permeability = Conv2D(1, (1, 1), activation='linear',
                          name='permeability')(conv9)
    poissonratio = Conv2D(1, (1, 1), activation='linear',
                          name='poisson_ratio')(conv9)
    porosity = Conv2D(1, (1, 1), activation='linear',
                      name='porosity')(conv9)
    shearimpedance = Conv2D(1, (1, 1), activation='linear',
                            name='shear_impedance')(conv9)
    shearmodulus = Conv2D(1, (1, 1), activation='linear',
                          name='shear_modulus')(conv9)
    vpvs = Conv2D(1, (1, 1), activation='linear',
                  name='VpVs')(conv9)
    youngmodulus = Conv2D(1, (1, 1), activation='linear',
                          name='youngs_modulus')(conv9)

    model = Model(inputs=[inputs],
                  outputs=[
                      acousticimpedance,
                      bulkmodulus,
                      density,
                      permeability,
                      poissonratio,
                      porosity,
                      shearimpedance,
                      shearmodulus,
                      vpvs,
                      youngmodulus
                  ])

    return model


num_channels = 3  # Set the number of input channels (offset images have 3 channels)
model = unet()
model.compile(optimizer=Adam(learning_rate=1e-3), loss=regularized_loss_masked,
              metrics=['MAE', r_squared, adjusted_r_squared])
model.summary()

history = model.fit(scaled_seismic,
                    [scaled_acousticimpedance,
                     scaled_bulk_modulus,
                     scaled_density,
                     scaled_permeability,
                     scaled_poissonratio,
                     scaled_porosity,
                     scaled_shear_impedance, scaled_shear_modulus,
                     scaled_Vp_Vs, scaled_Youngs_Modulus
                     ],
                    batch_size=8, epochs=60,
                    verbose=1, shuffle=True, validation_split=0.2)

# save history outputs to a csv file with same name as model
pd.DataFrame.from_dict(history.history).to_csv(output_dir + r"/model/history_model.csv", index=False)

# saving model
model.save(output_dir + r"/model")