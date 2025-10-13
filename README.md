# Concrete Patch Classification

This repository contains code for **texture classification of 3D concrete printing patches**, corresponding to the dataset described below.

## Dataset Reference
Our work focuses on the **texture classification patches**. This dataset provides three complementary resources:

1. **Reorganized version of the original 111 patches** with 5-fold splits.
2. **Extended set of 426 expert-annotated patches** including an additional geometric defect class (Crushed/Écrasé).  
3. **Synthetic patches generated with StyleGAN3**, covering all five classes.


## Training Configuration

To launch training, validation, or testing of the model, configure parameters such as dataset paths, number of epochs, optimizer settings, etc.  

### Example Configuration
  num_workers: Number of threads for data loading (0 = main thread).
  
  steps: Execution steps: train, prediction, evaluation. Can combine steps, e.g., "train,prediction".
  
  classes_names: Class names for classification.
  
  img_size: Input image size (224×224 pixels).
  
  no_of_epochs: Number of training epochs.
  
  batch_size: Batch size for training.
  
  desired_batchsize: Desired batch size for gradient accumulation.
  
  learning_rate: Learning rate.
  
  use_amp: Enable Automatic Mixed Precision (True/False).
  
  loss: initial = train from scratch; best = resume from previous weights.
  
  same_classes: Whether to treat classes as the same for fine-tuning scenarios.
  
  use_gpu: Enable GPU.
  
  forbid_augmentations: Disable online data augmentations.
  
  model_name: Model to use (e.g., efficientformerl3).
  
  seed: Random seed for reproducibility.
  
  extra_test_data: Use additional test data (True/False).
  
  prefetch_factor: Prefetch factor for dataloader batches.
  
  label_smooth: Label smoothing for classification regularization.
  
  weight_decay: Weight decay for optimizer.
  
  use_images_generees: Use generated images from StyleGAN3.
  
  num_genere: Number of generated images per class.
  
  Paths
  
  norm_params: Normalization parameters file.
  
  model_path: Pretrained model weights.
  
  prediction_path: Folder for predictions.
  
  evaluation_path: Folder for evaluation results.
  
  tb_path: TensorBoard events folder.
  
  log_path: Execution logs folder.
  
  images_generees_path: Folder with generated images.
  
  DataPaths
  
  train_image / val_image / test_image: Paths to training, validation, and test images.
  
  label_csv: CSV file containing image labels.

### Example Configuration

  ```ini
  [General]
  num_workers = 16
  experiment_name = data1fold2stylegan3new
  steps = train
  classes_names = Fluid,Good,Dry,Tearing
  img_size = 224
  no_of_epochs = 300
  batch_size = 32
  desired_batchsize = 32
  learning_rate = 0.0001
  use_amp = False
  loss = initial
  same_classes = False
  use_gpu = True
  forbid_augmentations = False
  model_name = efficientformerl3
  seed = 35
  extra_test_data = False
  prefetch_factor = 4
  label_smooth = 0.2
  weight_decay = 5e-3
  use_images_generees = True
  num_genere = 350,350,350,350
  
  [Paths]
  norm_params = ./
  model_path = ./pretrained/efficientformer_l3_300d.pth
  prediction_path = prediction
  evaluation_path = evaluation
  tb_path = events
  log_path = ./logs
  images_generees_path = ../data/out3
  
  [DataPaths]
  train_image = ../data/cross_validation(1erpatches)/fold_2/train
  val_image = ../data/cross_validation(1erpatches)/fold_2/val
  test_image = ../data/cross_validation(1erpatches)/fold_2/test
  label_csv = ../data/texture_windows-labels_imagesgenerees.csv

```


## Pretrained Models
The pretrained models used in this repository can be downloaded from:  
[Google Drive Folder](https://drive.google.com/file/d/1xeNe-H36_Me3ET4Yl3Nymxbsix0M8InG/view?usp=sharing)

### Included Pretrained Models

- **EfficientFormer-L3** – pretrained on ImageNet  
  [Model source / code](https://github.com/snap-research/EfficientFormer.git)  

- **InceptionResNetV2** – pretrained on ImageNet  
  > The pretrained weights are automatically downloaded from the official URL when first used.
  [Model source / code](https://github.com/Cadene/pretrained-models.pytorch.git) 

- **VGG19 (custom)** – pretrained on ImageNet  
  > The pretrained weights are automatically downloaded from the official URL when first used.


## Trained Models

The trained models in this repository can be downloaded from:

- [**Cross-validation on Sub-dataset 1 and Sub-dataset 3**](https://drive.google.com/file/d/15aNyjWIzbQUIV6rvJ2tOo9cstdn7bKef/view?usp=sharing)  
  **Sub-dataset 1:** Original annotated texture windows  
  - 111 labeled texture windows (width 200) extracted from 24 raw images  
  - Classes: Fluid, Good, Dry, Tearing  
  - 5-fold cross-validation  
  - Labels: `texture_windows-labels.csv`  
  **Sub-dataset 3:** Synthetic texture windows (StyleGAN3 generated)  
  - 1200 images per class (Fluid, Good, Dry, Tearing)  
  - Labels: `texture_windows-labels(111+stylegan3).csv`  

- [**Cross-validation on Sub-dataset 2 and Sub-dataset 3**](https://drive.google.com/file/d/15aNyjWIzbQUIV6rvJ2tOo9cstdn7bKef/view?usp=sharing)  
  **Sub-dataset 2:** Extended expert-annotated texture windows  
  - 426 labeled texture windows (width 200) extracted from 24 raw images  
  - Classes: Fluid, Good, Dry, Tearing, Crushed (Écrasé)  
  - 5-fold cross-validation  
  - Labels: `patch_labels(426extension).csv`  
  **Sub-dataset 3:** Synthetic texture windows (StyleGAN3 generated)  
  - 1200 images per class (Fluid, Good, Dry, Tearing, Crushed)  
  - Labels: `patch_labels(426extension+stylegan3).csv`  

