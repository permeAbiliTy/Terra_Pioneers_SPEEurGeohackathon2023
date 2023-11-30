#!/usr/bin/env python
# coding: utf-8

#    Team 'Terra Pioneers' entry to the SPE Europe Energy Geohackathon 2023 using model
#        to predict seismic inversion products on unseen prestack seismic data
#
#    Copyright (C) 2023  Team Terra Pioneers (Adam Turner, Mariam Shreif, Julien Kuhn de Chizelle, 
#                                                Ali Madani, and Saurav Bhattacharjee)
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Import the required libraries
import os
from pathlib import Path
import numpy as np
import segyio  # to read seismic
import tensorflow as tf
from empatches import EMPatches
from keras import backend as K
from keras import regularizers
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
# mid offset #
pathlist = Path(input_dir).glob('**/*_PreSTM_final_mid.sgy')
all_dataset_seismic_final_mid = []
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    with segyio.open(path_in_str, 'r', ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        all_dataset_seismic_final_mid.append(data)
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

unseen_seismic_dataset_far = all_dataset_seismic_final_far[2]  # keeping blind dataset
unseen_seismic_dataset_mid = all_dataset_seismic_final_mid[2]  # keeping blind dataset
unseen_seismic_dataset_near = all_dataset_seismic_final_near[2]  # keeping blind dataset 

seismic_dataset = np.concatenate([unseen_seismic_dataset_near[..., np.newaxis],
                                  unseen_seismic_dataset_mid[..., np.newaxis],
                                  unseen_seismic_dataset_far[..., np.newaxis]], axis=2)

print(np.shape(seismic_dataset))

# del all_dataset_seismic  # free up memory space
del all_dataset_seismic_final_near  # free up memory
del all_dataset_seismic_final_mid
del all_dataset_seismic_final_far

# Create an L2 regularizer
l2_regularizer = regularizers.l2(0.01)


# CUSTOM LOSS AND METRICS FUNCTIONS #3
def regularized_loss_masked(y_true, y_pred):
    # Calculate the Mean Squared Error
    loss = K.mean(K.square(y_pred * K.cast(y_true > tf.reduce_min(y_true), "float32") - y_true), axis=-1)

    # Add the L2 regularization
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            loss += l2_regularizer(layer.kernel)

    return loss


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


def r_squared(y_true, y_pred):  # Define a custom R-squared metric
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1 - ss_res / (ss_tot + tf.keras.backend.epsilon())


# LOADING THE DATASET #
model = tf.keras.models.load_model(output_dir + r"/model",
                                   custom_objects={"regularized_loss_masked": regularized_loss_masked,
                                                   "r_squared": r_squared,
                                                   "adjusted_r_squared": adjusted_r_squared})

# Opening the saved normalization files
with open('config.json', 'r') as f:
    config = json.load(f)

    # unseeen image #
img = np.transpose(seismic_dataset, (1, 0, 2))
img = (img - config['seismic'][0]) / config['seismic'][1]

print(np.shape(img))

# patch the image
emp = EMPatches()
img_patches, indices = emp.extract_patches(img, patchsize=256, overlap=0.1)
img_patches = np.concatenate(img_patches)
print(img_patches.shape[0])
img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256, 3])
img_patches = tf.expand_dims(img_patches, -1)
# img_patches.shape


# predict on image
prediction = model.predict(img_patches)

# merge patches back together
acoustic_impedance_result = ((emp.merge_patches(prediction[0], indices, mode='avg') * config['acoustic_impedance'][1])
                             + config['acoustic_impedance'][0])
bulk_modulus_result = ((emp.merge_patches(prediction[1], indices, mode='max') * config['bulk_modulus'][1])
                       + config['bulk_modulus'][0])
density_result = (emp.merge_patches(prediction[2], indices, mode='max') * config['density'][1]) + config['density'][0]
permeability_result = ((emp.merge_patches(prediction[3], indices, mode='max') * config['permeability'][1])
                       + config['permeability'][0])
poissonratio_result = ((emp.merge_patches(prediction[4], indices, mode='max') * config['poissonratio'][1])
                       + config['poissonratio'][0])
porosity_result = ((emp.merge_patches(prediction[5], indices, mode='max') * config['porosity'][1])
                   + config['porosity'][0])
shear_impedance_result = ((emp.merge_patches(prediction[6], indices, mode='max') * config['shear_impedance'][1])
                          + config['shear_impedance'][0])
shear_modulus_result = ((emp.merge_patches(prediction[7], indices, mode='max') * config['shear_modulus'][1])
                        + config['shear_modulus'][0])
VpVs_result = ((emp.merge_patches(prediction[8], indices, mode='max') * config['Vp_Vs'][1])
               + config['Vp_Vs'][0])
YoungsModulus_result = ((emp.merge_patches(prediction[9], indices, mode='max') * config['Youngs_Modulus'][1])
                        + config['Youngs_Modulus'][0])

# for acoustic impedance
output_file = output_dir + r'/L2EBN2020ASCAN025_acoustic_impedance.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(acoustic_impedance_result.T))

# for bulk modulus
output_file = output_dir + r'/L2EBN2020ASCAN025_bulk_density.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(bulk_modulus_result.T))

# for density
output_file = output_dir + r'/L2EBN2020ASCAN025_density.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(density_result.T))

# for permeability
output_file = output_dir + r'/L2EBN2020ASCAN025_permeability.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(permeability_result.T))

# for poissonratio
output_file = output_dir + r'/L2EBN2020ASCAN025_poisson_ratio.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(poissonratio_result.T))

# for porosity
output_file = output_dir + r'/L2EBN2020ASCAN025_porosity.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(porosity_result.T))

# for shear impedance
output_file = output_dir + r'/L2EBN2020ASCAN025_shear_impedance.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(shear_impedance_result.T))

# for shear mmodulus
output_file = output_dir + r'/L2EBN2020ASCAN025_shear_modulus.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(shear_modulus_result.T))

# for VpVs
output_file = output_dir + r'/L2EBN2020ASCAN025_VpVs.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(VpVs_result.T))

# for Youngs Modulus
output_file = output_dir + r'/L2EBN2020ASCAN025_Youngs_Modulus.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(YoungsModulus_result.T))
