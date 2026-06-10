# Concrete Patch Classification

This repository contains code for **texture classification of 3D concrete printing patches**, connecting to public dataset in [zenodo](https://doi.org/10.5281/zenodo.17078771) , corresponding to the dataset described below.

## Training Configuration

The **training**, **validation**, and **testing** processes are fully managed through the `configuration.ini` file.  
By editing this file, users can specify key parameters such as:

- Dataset paths
- Model architecture and pretrained weights
- Number of epochs and batch size
- Optimizer type and learning rate
- Cross-validation setup

Once the configuration is set, simply run the following command to start the process:

```bash
python main.py
```

## Parameters Configuration

The model execution is fully controlled via parameters defined in the configuration file (`configuration.ini`).  
Below is a summary of key parameters and their purpose, highlighting which ones are **required for training** and **required for testing**.

### General Parameters

- **steps**: Execution steps: `train`, `prediction`, `evaluation`. Multiple steps can be combined, e.g., `"train,prediction,evaluation"`.  
- **classes_names**: Class names for classification.(e.g. Fluid,Good,Dry,Tearing,Ecrase)  
- **img_size**: Input image size (e.g., 224/229).  
- **seed**: Random seed for reproducibility.  
- **use_gpu**: Enable GPU.  

### Training-Specific Parameters

These parameters **must be set for step:train**:

- **num_workers**: Number of threads for data loading (0 = main thread,the default num_workers=0 is used for testing).  
- **no_of_epochs**: Number of training epochs.  
- **batch_size**: Batch size for training.  
- **desired_batchsize**: Desired batch size for gradient accumulation.  
- **learning_rate**: Learning rate.  
- **use_amp**: Enable Automatic Mixed Precision (True/False).  
- **loss**: `"initial"` = train from scratch(default value:initial); `"best"` = resume from previous weights.  
- **same_classes**: Whether to treat classes as the same for fine-tuning scenarios(default value:False).  
- **forbid_augmentations**: Disable online data augmentations(default value:False).  
- **model_name**: Model to use (e.g., `efficientformerl3`).  
- **label_smooth**: Label smoothing for classification regularization.  
- **weight_decay**: Weight decay for optimizer.  
- **use_images_generees**: Use generated images from StyleGAN3.(If "True", parameter "num_genere" must be set and and this parameter should correspond to the parameter "classes_names".)  
- **num_genere**: Number of generated images per class.(e.g. 1200,1200,1200,1200）
- **images_generees_path**: Folder containing generated images (if `use_images_generees=True`).
- **model_path**: Path to pretrained model weights.
- **prefetch_factor**: Prefetch factor for dataloader batches(effective when num_workers>0).

### Testing-Specific Parameters

These parameters **must be set for step:prediction,evaluation**:

- **extra_test_data**: Use additional test data (true/false), which means it is not labeled(default value:False).  
- **model_path1~5**: 5 separate model weights for ensemble prediction. 
- **prediction_path**: Folder for saving predictions.  
- **evaluation_path**: Folder for evaluation results.  
- **tb_path**: TensorBoard events folder (optional).  
- **log_path**: Execution logs folder (optional).
- **norm_params**: Normalization parameters file for models which need images as input.
- **mean_features**= Normalization parameters file for models which need texture descripteur vector as input
- **std_features**= Normalization parameters file for models which need texture descripteur vector as input

### Data Paths (Required for Both Training and Testing)

- **train_image / val_image / test_image**: Paths to training, validation, and test images.  
- **label_csv**: CSV file containing image labels.  

**_Note: Only training specific parameters work in step train and only testing specific parameters work in step prediction/evaluation. Parameters that are irrelevant and unused in a step can be deleted_**



### Example of Training Configuration

  ```ini
  [General]

  num_workers = 16
  experiment_name = data1fold2stylegan3new
  steps = train
  classes_names = Fluid,Good,Dry,Tearing,Ecrase
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
  num_genere = 350,350,350,350,
  
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
  label_csv = ../data/texture_windows-labels_imagesgenerees.csv

```

### Example of Testing Configuration

  ```ini
  [General]
  experiment_name = fold2stylegan3
  steps = prediction,evaluation
  classes_names = Fluid,Good,Dry,Tearing,Ecrase
  img_size = 224
  batch_size = 1
  same_classes = False
  use_gpu = True
  model_name = texture_model
  extra_test_data = False


  [Paths]
  norm_params = ./test/norm_params.txt
  mean_features= ./test/mean_features_fold_1.npy
  std_features= ./test/std_features_fold_1.npy,
  model_path1=./test/model_epoch_004_loss_1.2211.pth	
  model_path2=./test/model_epoch_009_loss_1.2468.pth	
  model_path3=./test/model_epoch_014_loss_1.2547.pth
  model_path4=./test/model_epoch_016_loss_1.2008.pth	
  model_path5=./test/model_epoch_018_loss_1.2016.pth
  prediction_path = prediction
  evaluation_path = evaluation
  tb_path = events
  log_path = ./logs
  images_generees_path = ../data/out3
  
  [DataPaths]
  test_image = ../data/cross_validation(1erpatches)/fold_2/test
  label_csv = ../data/texture_windows-labels_imagesgenerees.csv

```


## Environment Setup

This project requires Python 3.8 and a set of specific packages for PyTorch-based training and evaluation.  
You can set up the environment either using **conda** or **pip**.

### Using Conda

The environment can be created from the included `environment.yaml`:

```bash
conda env create -f environment.yaml
conda activate 3dcp

```
### Using pip

Alternatively, install dependencies from `requirements.txt`:
```bash
conda create -n 3dcp python=3.8
conda activate 3dcp
pip install -r requirements.txt

```
**_Note: Some packages such as torch, torchaudio, and torchvision(commented in requirements.txt) may need to be installed from local wheel files for GPU/CUDA compatibility. Make sure to adjust the paths if needed. You can download local wheel files in [this page](https://download.pytorch.org/whl/torch_stable.html) (If the required version file cannot be found, try [this page](https://download.pytorch.org/whl/torch/))_**

## Pretrained Models
The pretrained models used in this repository can be downloaded from:  
[Google Drive Folder](https://drive.google.com/file/d/1xeNe-H36_Me3ET4Yl3Nymxbsix0M8InG/view?usp=sharing)

The pretrained backbones (Efficientformerl3, InceptionResNetV2, VGG19.) were initialized with weights trained on ImageNet, and subsequently fine-tuned on our texture datasets to adapt to the specific characteristics of 3D-printed concrete layer textures.

### Included Pretrained Models

- **EfficientFormer-L3** – pretrained on ImageNet  
  [Model source / code](https://github.com/snap-research/EfficientFormer.git)  

- **InceptionResNetV2** – pretrained on ImageNet  
  > The pretrained weights are automatically downloaded from the official URL when first used.
  [Model source / code](https://github.com/Cadene/pretrained-models.pytorch.git) 

- **VGG19 (custom)** – pretrained on ImageNet  
  > The pretrained weights are automatically downloaded from the official URL when first used.



## Generate an image using pkl weights in subdataset3.
[Download stylegan3 code](https://github.com/NVlabs/stylegan3.git)
```bash
python gen_images.py --outdir=out/fluid --trunc=1 --seeds=1-1200  --network=./model_final/fluidresume2721.pkl
python gen_images.py --outdir=out/good --trunc=1 --seeds=1-1200  --network=./model_final/goodresume2822.pkl
python gen_images.py --outdir=out/dry --trunc=1 --seeds=1-1200  --network=./model_final/dryresume1814.pkl
python gen_images.py --outdir=out/tearing --trunc=1 --seeds=1-1200  --network=./model_final/tearingresume504.pkl
python gen_images.py --outdir=out/ecrase --trunc=1 --seeds=1-1200 --class=4 --network=./network-snapshot-001501.pkl
```
### Parameters Explanation
- `--outdir` | Output directory path where generated images will be saved.|
- `--trunc` | Truncation value. Controls image diversity and fidelity: smaller values(0.5) produce more stable but less diverse images; larger values(2) increase variation but can reduce realism.(1 as defaut) |
- `--seeds` | Random seed or range of seeds to control the random latent vectors used for generation. Example: `1-1200` means seeds from 1 to 1200 will generate 1200 images. |
- `--network` | Path to the pretrained `.pkl` weight file. Each `.pkl` corresponds to a trained generator for a specific dataset or class.|
- `--class` *(optional)* | Used for conditional models. Specifies which class label to generate. Example: `--class=4` means generate images of class index 4, which means class "ecrase". The index of five classes are (fluid,good,dry,tearing,ecrase). The first four models are dedicated to generating 1 specific class(fluid,good,dry,tearing), but network-snapshot-001501.pkl can generate 5 classes. You can specify a specific class by `--class=4`, which is generally used for class "ecrase"|
