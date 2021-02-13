# Unsupervised-Anomaly-Detection-with-SSIM-AE

## Story on Medium

[https://medium.com/@majpaw1996/anomaly-detection-in-computer-vision-with-ssim-ae-2d5256ffc06b] Anomaly Detection in Computer Vision with SSIM-AE

## Related Works

[1] Bergmann, Paul, et al. "Improving unsupervised defect segmentation by applying structural similarity to autoencoders." arXiv preprint arXiv:1807.02011 (2018)

[2] Bergmann, Paul, et al. "MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019

## Data

Used datasets are available on the websites:
- *grid* and *carpet*: https://www.mvtec.com/company/research/datasets/mvtec-ad/,
- *woven fabrics* (*texture_1* and *texture_2*): https://www.mvtec.com/company/research/publications/

Put downloaded datasets in directory *data/*

## Training

python AE_training.py 
[-h] [--dataset_name DATASET_NAME] [--latent_dim LATENT_DIM]
[--batch_size BATCH_SIZE] [--training_loss TRAINING_LOSS]
[--load_model LOAD_MODEL] [--random_crop RANDOM_CROP]

**Parameters**:
- dataset_name (name of dataset used for training) e.g. "grid", "carpet", "texture_1", "texture_2",
- latent_dim (dimension of bottleneck in autoencoder architecture) e.g. 100,
- batch_size (batch size used for autoencoder training) e.g. 8,
- training_loss (loss used for autoencoder training): "ssim" or "mse",
- load_model (load weights of trained model): 1 or 0,
- random_crop (random crop 10k ROIs of size 128): 1 or 0.

## Evaluation

python AE_evaluation.py
