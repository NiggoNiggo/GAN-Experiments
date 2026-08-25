### This Repository implements a historical overview of *Generative Adversarial Networks*
This project provides a collection of implementations of various Generative Adversarial Network (GAN) architectures, loss functions, and regularization techniques. It can be viewed as a methodological journey through the development of GANs, starting with the original Vanilla GAN and progressing to more advanced architectures such as DCGAN and WGAN, with the eventual goal of exploring modern state-of-the-art approaches such as StyleGAN.

The theoretical foundations and key concepts are documented in the \textit{foundation} directory, where the underlying methods and their respective principles are summarized and explained. A dedicated summary chapter provides a comparative analysis of the implemented approaches.

The primary objective of the project is to systematically investigate the interaction between different GAN architectures, loss functions, and regularization techniques, with a particular focus on identifying combinations that provide stable and reliable training.
Unfortunately, the project is ongoing, therefore the READme is not up to data and will only be updated randomly. Please find the description and interpretation of results in the pdf file contained in the foundation folder.



### Structure

## Project Structure

```text
project/
├── loss_functions/
├── architectures/
├── core/
├── data/
├── observer/
├── organization/
├── param_configs/
├── plotting/
├── regularisations/
├── testing/
├── training/
├── utils/
└── validation/

```
# The project is structured as follows:

## 1. `loss_functions`

Contains various loss functions, e.g.:

- WGAN
- Vanilla GAN
- LSGAN
- Hinge Loss

## 2. `architectures`

Contains various GAN architectures.

- Each file contains the corresponding discriminator and generator.
- `layers.py` contains various layers, including:
  - ResNet layers
  - Upsampling and downsampling layers
  - `ConvTranspose`
  - `Upsample`
- `init_weights.py` contains various weight initialization techniques for setting the model parameters at the beginning of training.

## 3. `core`

Contains core functionality, including:

- Setting optimizers from a generic configuration
- Registry pattern for dynamically initializing models
- Loading training configurations from a single `.yaml` file

## 4. `data`

Contains various datasets for benchmark datasets.

- Benchmark datasets are currently still under development.
- `wrapper.py` provides a wrapper dataset that allows datasets to work with or without labels.

## 5. `observer`

Contains functionality for monitoring the training process.

- Evaluating models
- Generating plots
- Saving models
- Performing tasks that are regularly executed after a certain number of epochs

> **Note:** This part of the project is currently still somewhat unstructured.

## 6. `organization`

Contains helper functions for organizing the project and maintaining a structured project layout.

## 7. `param_configs`

Contains the configuration files for each GAN architecture.

- Each GAN architecture has its own `.yaml` configuration file.
- The structure and usage of these configuration files are explained later.

## 8. `plotting`

Contains files and functions for plotting and visualizing results.

## 9. `regularisations`

Contains regularization techniques for improving the stability of GAN training.

## 10. `testing`

Contains tests for the project.

> **Note:** Testing functionality is currently under development.

## 11. `training`

Contains a generic abstract training class that can be extended for different training procedures.

## 12. `utils`

Contains general helper functions used throughout the project.

> **Note:** This part of the project is currently still under development.

## 13. `validation`

Contains validation metrics for evaluating GAN performance, including:

- KID (Kernel Inception Distance)
- FID (Fréchet Inception Distance)
- IS (Inception Score)


### Training a GAN 
Fill in the yaml file in the folder `params_config`. Afterwards run main.py
Text is coming soon. Need to be found.



    









### Implemented Papers:
- [Generative Adversarial Networks (GAN)](https://arxiv.org/abs/1406.2661)
- [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (DCGAN)](https://arxiv.org/abs/1511.06434)
- [Least Squares Generative Adversarial Networks (LSGAN)](https://arxiv.org/abs/1611.04076)
- [Wasserstein GAN (WGAN)](https://arxiv.org/abs/1701.07875)
- [Improved Training of Wasserstein GANs (WGAN GP)](https://arxiv.org/abs/1704.00028)
- [Generative Adversarial Network based on Resnet for Conditional Image Restoration](https://arxiv.org/abs/1707.04881)
- [Conditional Generative Adversarial Nets (cGAN)](https://arxiv.org/abs/1411.1784)
- [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957)
- [Improved Techniques for Training GANs](https://arxiv.org/abs/1606.03498)