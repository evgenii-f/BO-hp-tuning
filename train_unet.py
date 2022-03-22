import numpy as np
import os
import tensorflow
#from focal_loss import BinaryFocalLoss
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, EarlyStopping, TensorBoard
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
import random
from PIL import Image
import sys
import time

def fp(y_true, y_pred):
    return tensorflow.math.count_nonzero(tensorflow.math.multiply(
        tensorflow.convert_to_tensor(tensorflow.math.round(y_pred), dtype=tensorflow.float32), (y_true - 1)
    ))

def fn(y_true, y_pred):
    return tensorflow.math.count_nonzero(tensorflow.math.multiply(
        (tensorflow.convert_to_tensor(tensorflow.math.round(y_pred), dtype=tensorflow.float32) - 1), y_true
    ))

def tp(y_true, y_pred):
    return tensorflow.math.count_nonzero(tensorflow.math.multiply(
        y_true, tensorflow.convert_to_tensor(tensorflow.math.round(y_pred), dtype=tensorflow.float32)
    ))

def recall2(y_true, y_pred):
    TP = tp(y_true, y_pred)
    FN = fn(y_true, y_pred)
    recall = tensorflow.math.divide(TP, (TP + FN))
    return recall

def precision2(y_true, y_pred):
    TP = tp(y_true, y_pred)
    FP = fp(y_true, y_pred)
    precision = tensorflow.math.divide(TP, (TP + FP))
    return precision

def dice_coefficient_metric(y_true, y_pred):
    re = recall2(y_true, y_pred)
    pr = precision2(y_true, y_pred)
    numerator = 2*re*pr
    denominator = re+pr
    dice = numerator / denominator
    return dice


def dice_coefficient(y_true, y_pred):
    y_pred = tensorflow.convert_to_tensor(y_pred, dtype=tensorflow.float32)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    numerator = (2 * K.sum(y_true * y_pred)) + 1.
    denominator = (K.sum(K.square(y_true)) + K.sum(K.square(y_pred))) + 1.
    dice = numerator / denominator
    return dice


def dice_loss(y_true, y_pred):
    dice = dice_coefficient(y_true, y_pred)
    loss = 1. - dice
    return loss


class CustomDiceLoss(tensorflow.keras.losses.Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        return dice_loss(y_true=y_true, y_pred=y_pred)


def conv_block(x, filter_size, size, dropout, batch_norm=False):
    conv = Conv2D(size, (filter_size, filter_size), padding="same")(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)

    conv = Conv2D(size, (filter_size, filter_size), padding="same")(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=3)(conv)
    conv = Activation("relu")(conv)
    
    if dropout > 0:
        conv = Dropout(dropout)(conv)

    return conv


def unet(image_shape=(512,512,3), dropout_rate=0.0, batch_norm=True):
    # network structure
    FILTER_NUM = 64 # number of filters for the first layer
    FILTER_SIZE = 3 # size of the convolutional filter
    UP_SAMP_SIZE = 2 # size of upsampling filters
    
    inputs = Input(image_shape)
    # Downsampling 1
    conv_128 = conv_block(inputs, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)
    pool_64 = MaxPooling2D(pool_size=(2,2))(conv_128)
    # Downsampling 2
    conv_64 = conv_block(pool_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    pool_32 = MaxPooling2D(pool_size=(2,2))(conv_64)
    # Downsampling 3
    conv_32 = conv_block(pool_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    pool_16 = MaxPooling2D(pool_size=(2,2))(conv_32)
    # Downsampling 4
    conv_16 = conv_block(pool_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    pool_8 = MaxPooling2D(pool_size=(2,2))(conv_16)
    # Downsampling 5
    conv_8 = conv_block(pool_8, FILTER_SIZE, 16*FILTER_NUM, dropout_rate, batch_norm)

    # Upsampling layers
   
    up_16 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(conv_8)
    up_16 = concatenate([up_16, conv_16], axis=3)
    up_conv_16 = conv_block(up_16, FILTER_SIZE, 8*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 7
    
    up_32 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_16)
    up_32 = concatenate([up_32, conv_32], axis=3)
    up_conv_32 = conv_block(up_32, FILTER_SIZE, 4*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 8
    
    up_64 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_32)
    up_64 = concatenate([up_64, conv_64], axis=3)
    up_conv_64 = conv_block(up_64, FILTER_SIZE, 2*FILTER_NUM, dropout_rate, batch_norm)
    # UpRes 9
   
    up_128 = UpSampling2D(size=(UP_SAMP_SIZE, UP_SAMP_SIZE), data_format="channels_last")(up_conv_64)
    up_128 = concatenate([up_128, conv_128], axis=3)
    up_conv_128 = conv_block(up_128, FILTER_SIZE, FILTER_NUM, dropout_rate, batch_norm)

    # 1*1 convolutional layers
   
    conv_final = Conv2D(1, kernel_size=(1,1))(up_conv_128)
    conv_final = BatchNormalization(axis=3)(conv_final)
    conv_final = Activation('sigmoid')(conv_final)

    # Model 
    model = Model(inputs, conv_final, name="UNet")
    #print(model.summary())
    return model


def train_unet(datapath, checkpoint_dir, hyperparameters, gpu=None):
    train_data = np.load(os.path.join(datapath, 'train.npz'))
    x_train = train_data['arr_0']
    y_train = train_data['arr_1']
    y_train = y_train.astype('float32')
    val_data = np.load(os.path.join(datapath, 'val.npz'))
    x_val = val_data['arr_0']
    y_val = val_data['arr_1']
    y_val = y_val.astype('float32')

    IMAGE_SHAPE = hyperparameters['image_shape']
    EPOCHS = hyperparameters['epochs']
    BATCH_NORM = hyperparameters['batch_norm']
    BATCH_SIZE = hyperparameters['batch_size']
    STEPS_PER_EPOCH = len(x_train) // BATCH_SIZE
    LEARNING_RATE = hyperparameters['learning_rate']
    DROPOUT_RATE = hyperparameters['dropout_rate']

    if gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    
    model = unet(image_shape=IMAGE_SHAPE, dropout_rate=DROPOUT_RATE, batch_norm=BATCH_NORM)
    opt = Adam(learning_rate=LEARNING_RATE)

    model.compile(optimizer=opt, loss=CustomDiceLoss(), metrics=[tensorflow.keras.metrics.Recall(),
                    tensorflow.keras.metrics.Precision(), recall2, precision2, dice_coefficient_metric]
                  )

    csv_logger = tensorflow.keras.callbacks.CSVLogger(filename=checkpoint_dir + 'history_log.csv', append=False)
    tensorboard_callback = TensorBoard(log_dir= os.path.join(checkpoint_dir, 'logs'))
    earlyStopping_callback = EarlyStopping(monitor='val_loss', patience=12)
    lr_reduce_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint(os.path.join(checkpoint_dir, 'unet_model.h5'),
                                       monitor='val_dice_coefficient_metric', verbose=1)
    
    model_history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,
                              steps_per_epoch=STEPS_PER_EPOCH, validation_data=(x_val, y_val),
                              callbacks=[csv_logger, tensorboard_callback, earlyStopping_callback,
                                         lr_reduce_callback, model_checkpoint]
                              )

    return model_history
