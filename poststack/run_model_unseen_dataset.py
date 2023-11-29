#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import numpy as np
# import pandas as pd
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

# importing input training dataset without AGC
pathlist = Path(input_dir).glob('**/*_full.sgy')
all_dataset_seismic = []
segypath = []  # saving the path for the data for later on to create a new segy with rock properties
for path in pathlist:  # iterating through the list of seismic data
    # because path is object not string
    path_in_str = str(path)
    segypath.append(path_in_str)
    # print(path_in_str)
    with segyio.open(path_in_str, 'r', ignore_geometry=True) as segyfile:
        data = segyfile.trace.raw[:]
        all_dataset_seismic.append(data)

# seismic_dataset = [all_dataset_seismic[i] for i in [0, 1, 3, 4, 5, 6, 7]]
unseen_seismic_dataset = all_dataset_seismic[2]

del all_dataset_seismic

# Create an L2 regularizer
l2_regularizer = regularizers.l2(0.01)


def regularized_loss_masked(y_true, y_pred):
    # Calculate the Mean Squared Error, masking zero values in ground truth
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


# Load model_old
model = tf.keras.models.load_model(output_dir + r"/model_old",
                                   custom_objects={"regularized_loss_masked": regularized_loss_masked,
                                                   "r_squared": r_squared,
                                                   "adjusted_r_squared": adjusted_r_squared})

# Open config file
with open('config.json', 'r') as f:
    config = json.load(f)

# Rescale unseen dataset
img = unseen_seismic_dataset.T
img = (img - config['seismic'][0]) / config['seismic'][1]

# Patch unseen dataset
emp = EMPatches()
img_patches, indices = emp.extract_patches(img, patchsize=256, overlap=0.1)
img_patches = np.concatenate(img_patches)
img_patches = tf.reshape(img_patches, [int(img_patches.shape[0] / 256), 256, 256])
img_patches = tf.expand_dims(img_patches, -1)
# img_patches.shape

# Predict the outputs on the unseen dataset
prediction = model.predict(img_patches)

# Scale outputs back to original values
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

input_file = segypath[2]  # path of the unseen seismic line

# save acoustic impedance as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_acoustic_impedance.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(acoustic_impedance_result.T))

# save bulk modulus as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_bulk_density.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(bulk_modulus_result.T))

# save density as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_density.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(density_result.T))

# save permeability as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_permeability.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(permeability_result.T))

# save poissonratio as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_poisson_ratio.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(poissonratio_result.T))

# save bulk modulus as segy
output_file = output_dir + r'/L2EBN2020ASCAN025_porosity.sgy'
segyio.tools.from_array2D(output_file, np.squeeze(porosity_result.T))
