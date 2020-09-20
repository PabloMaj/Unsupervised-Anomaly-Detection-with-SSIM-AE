import cv2
import os
import numpy as np
import random
import warnings
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, LeakyReLU
from tqdm import tqdm
from sklearn.model_selection import train_test_split

warnings.simplefilter("ignore")


def architecture_MVTEC(input_shape=(128, 128, 1), latent_dim=100):

    parameters = dict()
    N_layers = 9
    parameters["filters"] = [32, 32, 32, 64, 64, 128, 64, 32, latent_dim]
    parameters["kernel_size"] = [4, 4, 3, 4, 3, 4, 3, 3, 8]
    parameters["strides"] = [2, 2, 1, 2, 1, 2, 1, 1, 1]
    parameters["padding"] = ["same" for _ in range(N_layers-1)] + ["valid"]

    # Input
    inputs = Input(shape=input_shape)
    x = inputs

    # Encoder
    for i in range(0, N_layers):
        x = Conv2D(
          filters=parameters["filters"][i],
          kernel_size=parameters["kernel_size"][i],
          strides=parameters["strides"][i],
          padding=parameters["padding"][i])(x)
        x = LeakyReLU(alpha=0.2)(x)

    # Decoder
    for i in reversed(range(0, N_layers)):
        x = Conv2DTranspose(
          filters=parameters["filters"][i],
          kernel_size=parameters["kernel_size"][i],
          strides=parameters["strides"][i],
          padding=parameters["padding"][i])(x)
        x = LeakyReLU(alpha=0.2)(x)

    # Output
    x = Conv2DTranspose(
      filters=input_shape[2],
      kernel_size=(3, 3),
      strides=(1, 1),
      padding="same")(x)

    outputs = x

    # Autoencoder
    autoencoder = Model(inputs, outputs, name="autoencoder")

    return autoencoder


def read_data(dataset_name="carpet"):

    path_to_train_set = f"data/{dataset_name}/train/"
    path_to_test_set = f"data/{dataset_name}/test/"

    train_data = dict()
    test_data = dict()

    ROI_extracted_size = (256, 256)
    ROI_resized_size = (128, 128)

    for path_to_data, data_dict in [(path_to_train_set, train_data), (path_to_test_set, test_data)]:
        for category in tqdm(os.listdir(path_to_data)):
            for img_name in tqdm(os.listdir(path_to_data + str(category))):

                img = cv2.imread(f"{path_to_data}{category}/{img_name}", 0)
                for x_start in range(0, img.shape[0]-ROI_extracted_size[0]+1, ROI_extracted_size[0]):
                    for y_start in range(0, img.shape[1]-ROI_extracted_size[1]+1, ROI_extracted_size[1]):

                        x_end = x_start + ROI_extracted_size[0]
                        y_end = y_start + ROI_extracted_size[1]

                        img_ROI = img[x_start:x_end, y_start:y_end]
                        img_resized = cv2.resize(img_ROI, ROI_resized_size)
                        img_resized = img_resized.astype("float32") / 255.0

                        if category not in data_dict:
                            data_dict[category] = dict()
                        if img_name not in data_dict[category]:
                            data_dict[category][img_name] = []
                        data_dict[category][img_name].append(img_resized)

    return train_data, test_data


def read_data_with_random_crop(dataset_name="carpet", N_train=10**4):

    path_to_train_set = f"data/{dataset_name}/train/"

    if dataset_name in ["texture_1", "texture_2"]:
        img_resized_size = (256, 256)
        crop_size = (128, 128)
    elif dataset_name in ["carpet", "grid"]:
        img_resized_size = (512, 512)
        crop_size = (128, 128)

    train_data = []

    for i in tqdm(range(0, N_train)):

        img_name = random.choice(list(os.listdir(path_to_train_set + "good")))
        img = cv2.imread(f"{path_to_train_set}good/{img_name}", 0)
        img_resized = cv2.resize(img, img_resized_size)
        x_start = random.randint(0, img_resized.shape[0]-crop_size[0])
        y_start = random.randint(0, img_resized.shape[1]-crop_size[1])
        x_end = x_start + crop_size[0]
        y_end = y_start + crop_size[1]
        crop = img_resized[x_start:x_end, y_start:y_end]
        crop = crop.astype("float32") / 255.0
        train_data.append(crop)

    X_train = np.array(train_data)

    return X_train


def DSSIM_loss(y_true, y_pred):
    return 1/2 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0))/2


def train_model(input_shape=(128, 128, 1), dataset_name="carpet", latent_dim=100, training_loss="ssim", load_model=True, random_crop=False, batch_size=8):

    # 1) read data
    if random_crop:
        X_train = read_data_with_random_crop(dataset_name=dataset_name, N_train=10000)
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

    # 2) create autoencoder
    autoencoder = architecture_MVTEC(input_shape=input_shape, latent_dim=latent_dim)
    autoencoder.summary()

    # 3) choose training_loss function and optimizer
    opt = Adam(learning_rate=2*10e-5)

    if training_loss == "mse":
        autoencoder.compile(loss="mse", optimizer=opt)
    elif training_loss == "ssim":
        autoencoder.compile(loss=DSSIM_loss, optimizer=opt)

    # 4) set callbacks

    if not os.path.exists(f"model_weights/{dataset_name}/"):
        os.mkdir(f"model_weights/{dataset_name}/")

    path_to_save_model = f"model_weights/{dataset_name}/"
    name = f"a_{latent_dim}_loss_{training_loss}_batch_{batch_size}.hdf5"
    path_to_save_model += name

    model_checkpoint_callback = ModelCheckpoint(
      save_weights_only=False, filepath=path_to_save_model,
      monitor='val_loss', save_best_only=True)
    early_stopping_callback = EarlyStopping(
      monitor='val_loss', min_delta=0, patience=50, verbose=0, mode='auto')

    # 5) load weights and training model
    if load_model:
        if name in os.listdir(f"model_weights/{dataset_name}"):
            autoencoder.load_weights(path_to_save_model)

    autoencoder.fit(
      x=X_train, y=Y_train,
      epochs=200, batch_size=batch_size,
      shuffle=True, validation_data=(X_val, Y_val),
      callbacks=[model_checkpoint_callback, early_stopping_callback])


def parse_args():
    parser = argparse.ArgumentParser('AE_SSIM')
    parser.add_argument("--dataset_name", type=str, default="carpet")
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--training_loss", type=str, default="ssim")
    parser.add_argument("--load_model", type=bool, default=0)
    parser.add_argument("--random_crop", type=bool, default=1)
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    print(
     f"\ndataset_name={args.dataset_name}\ntraining_loss={args.training_loss}\n"
     f"latent_dim={args.latent_dim}\nbatch_size={args.batch_size}\n"
     f"load_model={args.load_model}\nrandom_crop={args.random_crop}")

    train_model(
     input_shape=(128, 128, 1), dataset_name=args.dataset_name,
     latent_dim=args.latent_dim, training_loss=args.training_loss,
     load_model=args.load_model, random_crop=args.random_crop,
     batch_size=args.batch_size)
