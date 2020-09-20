# Unsupervised-Anomaly-Detection-with-SSIM-AE

## Related Works

[1] Bergmann, Paul, et al. "Improving unsupervised defect segmentation by applying structural similarity to autoencoders." arXiv preprint arXiv:1807.02011 (2018)

[2] Bergmann, Paul, et al. "MVTec AD--A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2019

## Data

Used datasets are available on the websites:
- *grid* and *carpet*: https://www.mvtec.com/company/research/datasets/mvtec-ad/,
- *woven fabrics* (*texture_1* and *texture_2*): https://www.mvtec.com/company/research/publications/

## Training

python AE_training.py 
[-h] [--dataset_name DATASET_NAME] [--latent_dim LATENT_DIM]
[--batch_size BATCH_SIZE] [--training_loss TRAINING_LOSS]
[--load_model LOAD_MODEL] [--random_crop RANDOM_CROP]

## Evaluation
