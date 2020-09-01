import cv2
import os
from tqdm import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
warnings.simplefilter("ignore")

from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
import tensorflow as tf

def architecture_MVTEC(input_shape=(128, 128, 1), latent_dim=100):

  #-----------------------------------
  #encoder

  #Input
  inputs = Input(shape=input_shape)
  x = inputs

  #Conv1 (encoder)
  x = Conv2D(filters = 32, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv2 (encoder)
  x = Conv2D(filters = 32, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv3 (encoder)
  x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv4 (encoder)
  x = Conv2D(filters = 64, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv5 (encoder)
  x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv6 (encoder)
  x = Conv2D(filters = 128, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv7 (encoder)
  x = Conv2D(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv8 (encoder)
  x = Conv2D(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv9 (encoder)
  x = Conv2D(filters = latent_dim, kernel_size = (8, 8), strides = (1,1), padding = "valid")(x)
  x = LeakyReLU(alpha=0.2)(x)  

  #-----------------------------------
  #decoder

  #Conv9 (decoder)
  x = Conv2DTranspose(filters = latent_dim, kernel_size = (8, 8), strides = (1,1), padding = "valid")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv8 (decoder)
  x = Conv2DTranspose(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv7 (decoder)
  x = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides=(1,1), padding="same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv6 (decoder)
  x = Conv2DTranspose(filters = 128, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv5 (decoder)
  x = Conv2DTranspose(filters = 64, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv4 (decoder)
  x = Conv2DTranspose(filters = 64, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv3 (decoder)
  x = Conv2DTranspose(filters = 32, kernel_size = (3, 3), strides = (1,1), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv2 (decoder)
  x = Conv2DTranspose(filters = 32, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  #Conv1 (decoder)
  x = Conv2DTranspose(filters = 32, kernel_size = (4, 4), strides = (2,2), padding = "same")(x)
  x = LeakyReLU(alpha=0.2)(x)

  x = Conv2DTranspose(filters = input_shape[2],  kernel_size = (3, 3), strides=(1,1), padding="same")(x)
  outputs = x
  
  #-----------------------------------

  #-----------------------------------
  #autoencoder
  autoencoder = Model(inputs, outputs, name="autoencoder")
  #-----------------------------------

  return autoencoder

def read_data(dataset_name = "carpet"):

  path_to_train_set = f"data/{dataset_name}/train/"
  path_to_test_set =  f"data/{dataset_name}/test/"

  train_data = dict()
  test_data  = dict()

  ROI_extracted_size = (256,256)
  ROI_resized_size = (128,128)

  for path_to_data , data_dict in [(path_to_train_set,train_data),(path_to_test_set,test_data)]:
    for category in tqdm(os.listdir(path_to_data)):
      for img_name in tqdm(os.listdir(path_to_data + str(category))):

        img = cv2.imread(f"{path_to_data}{category}/{img_name}",0)
        
        for x_start in range(0,img.shape[0]-ROI_extracted_size[0]+1,ROI_extracted_size[0]):
          for y_start in range(0,img.shape[1]-ROI_extracted_size[1]+1,ROI_extracted_size[1]):

            x_end = x_start + ROI_extracted_size[0]
            y_end = y_start + ROI_extracted_size[1]

            img_ROI = img[x_start:x_end,y_start:y_end]
            img_resized = cv2.resize(img_ROI, ROI_resized_size)
            img_resized = img_resized.astype("float32") / 255.0

            if category not in data_dict:
              data_dict[category] = dict()
            if img_name not in data_dict[category]:
              data_dict[category][img_name] = []
            data_dict[category][img_name].append(img_resized)
  
  return train_data, test_data

def read_data_with_random_crop(dataset_name = "carpet",N_train = 10**4):

  path_to_train_set = f"data/{dataset_name}/train/"
  
  ROI_extracted_size = (256,256)
  ROI_resized_size = (128,128)

  train_data = []
  
  for i in tqdm(range(0,N_train)):
    
    img_name = random.choice(list(os.listdir(path_to_train_set + "good")))
    img = cv2.imread(f"{path_to_train_set}good/{img_name}",0)
    x_start = random.randint(0, img.shape[0]-ROI_extracted_size[0])
    y_start = random.randint(0, img.shape[1]-ROI_extracted_size[1])
    x_end = x_start + ROI_extracted_size[0]
    y_end = y_start + ROI_extracted_size[1]

    img_ROI = img[x_start:x_end,y_start:y_end]
    img_resized = cv2.resize(img_ROI, ROI_resized_size)
    img_resized = img_resized.astype("float32") / 255.0

    train_data.append(img_resized)

  X_train = np.array(train_data)

  return X_train

def SSIMLoss(y_true, y_pred):
  return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))

def train_model(input_shape = (128,128,1), dataset_name = "carpet" ,latent_dim = 100, training_loss = "ssim", load_model = True, random_crop = False, batch_size = 8):

  #1) read data
  if random_crop==True:
    X_train = read_data_with_random_crop(dataset_name = dataset_name, N_train = 10000)
    X_train = np.expand_dims(X_train, axis=-1)
  else:
    train_data, _ = read_data(dataset_name=dataset_name)
    X_train = []
    for img_name in train_data['good'].keys():
      for img in train_data['good'][img_name]:
        X_train.append(img)
    X_train = np.array(X_train)
    X_train = np.expand_dims(X_train, axis=-1)

  Y_train = X_train
  X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

  #2) create autoencoder
  autoencoder = architecture_MVTEC(input_shape=input_shape,latent_dim=latent_dim)
  autoencoder.summary()

  #3) choose training_loss function and optimizer
  opt = Adam(learning_rate = 2*10e-5)

  if training_loss == "mse":
    autoencoder.compile(loss="mse", optimizer=opt)
  elif training_loss == "ssim":
    autoencoder.compile(loss=SSIMLoss, optimizer=opt)


  #4) path to save model, callbacks, read weights
  path_to_save_model = f"model_weights/{dataset_name}/"
  name = f"a_{latent_dim}_loss_{training_loss}_batch_{batch_size}.hdf5"
  path_to_save_model += name

  model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(save_weights_only=False,filepath=path_to_save_model,monitor='val_loss',save_best_only=True)
  early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=100, verbose=0, mode='auto')

  #5) training model  
  if load_model:
    if name in os.listdir(f"model_weights/{dataset_name}"):
      autoencoder.load_weights(path_to_save_model)
      autoencoder.fit(X_train, Y_train, epochs=300, batch_size=batch_size, shuffle=True, validation_data=(X_val, Y_val),callbacks=[model_checkpoint_callback,early_stopping_callback])
  else:
    autoencoder.fit(X_train, Y_train, epochs=300, batch_size=batch_size, shuffle=True, validation_data=(X_val, Y_val),callbacks=[model_checkpoint_callback,early_stopping_callback])

if __name__ == "__main__":
  input_shape  = (128,128,1) 
  dataset_name = "woven_fabric_2"
  latent_dim = 100
  batch_size = 8
  training_loss = "ssim"
  load_model = False
  random_crop = True

  print(f"\n dataset_name={dataset_name}\n training_loss={training_loss}\n latent_dim={latent_dim}\n batch_size={batch_size}\n")
  train_model(input_shape = input_shape,dataset_name = dataset_name,latent_dim = latent_dim,training_loss = training_loss,load_model = False,random_crop=random_crop, batch_size=batch_size)