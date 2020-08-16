# DeepSegment
Code repository for our manuscript on training CNN classifiers for semantic segmentation using partially annotated images.

Part of the code in this repository was developed with heavy reference to another repository on segmentation of cardiac MR images: <https://github.com/baumgach/acdc_segmenter>

## Getting started
Clone this repository to your target machine. Throughout this guide, ```$DEEPSEGMENT_DIR``` represents where this repository resides on your machine.

To install required python packages, execute the command:

```python $DEEPSEGMENT_DIR/requirements.txt```

## Data

### download NIfTIs
Head to the Medical Segmentation Decathlon site (http://medicaldecathlon.com/) and download the required datasets. Alternatively, direct links to the dataset are listed below for convenience, which is correct at the time of writing:
- Task03_Liver: https://drive.google.com/uc?export=download&id=1jyVGUGyxKBXV6_9ivuZapQS8eUJXCIpu
- Task07_Pancreas: https://drive.google.com/uc?export=download&id=1YZQFSonulXuagMIfbJkZeTFJ6qEUuUxL
- Task09_Spleen: https://drive.google.com/uc?export=download&id=1jzeNU1EKnK81PyTsrx0ujfNl-t0Jo8uE

Place the downloaded tar files under ```$DEEPSEGMENT_DIR/data/MSD2018/scripts/```. Then, perform either of the following steps:
- execute the command ```python $DEEPSEGMENT_DIR/data/MSD2018/scripts/extract_nii.py```, OR
- manually unpack the tar files to the parent directory, followed by removing all system/hidden files if any (i.e. those that begin with '.' in their filename)

### train/test split
The NIfTI images have to be splitted into train and test sets. Execute the command:

```python $DEEPSEGMENT_DIR/projects/MSD_full/data/scripts/make_symlink.py```

to split the images in each dataset into train and test sets by the ratio 80:20. This creates multiple directories under ```$DEEPSEGMENT_DIR/projects/MSD_full/data/nii/```, each containing symbolic links pointing toward the NIfTI images at ```$DEEPSEGMENT_DIR/data/MSD2018/```

### preparing HDF5s
To speed up model training, the NIfTI images have to be pre-processed and bundled into HDF5 files. Execute the command:

```python $DEEPSEGMENT_DIR/projects/MSD_full/data/scripts/build_hdf5.py```

to generate ```.hdf5``` files containing axial, coronal, or sagittal slices extracted and pre-processed from the NIfTI images. Pre-processing steps taken in the script can be summarized as follows:
- resample the 3D NIfTI images to have 1.5x1.5mm slice resolution and 3.5mm slice thickness
- slice the resampled 3D images in axial, coronal, or sagittal directions
- crop or pad the 2D slices to 256x256 pixels
- for slices extracted from the label images, modify pixel value to follow the new label-class-to-value scheme:
    - background: 0 (no mapping required)
    - liver: 1 (no mapping required)
    - liver_tumor: 2 (no mapping required)
    - pancreas: 3 (mapped; originally 1)
    - pancreas_tumor: 4 (mapped; originally 2)
    - spleen: 5 (mapped; originally 1)

### repeat for smaller datasets
Second part of our manuscript utilized subsets of images from the original datasets. Execute the command:

```python $DEEPSEGMENT_DIR/projects/MSD_partial/data/scripts/make_symlink_partial.py```

to first split the images in each dataset into train and test sets. Instead of using 80 percent of the full dataset as train set, this will generate train sets at 2.5, 5, 10, 20, and 40 percent of the full dataset. For all train percentages, the same test set from the 80:20 split will be used. Then, execute the command:

```python $DEEPSEGMENT_DIR/projects/MSD_partial/data/scripts/build_hdf5_partial.py```

to generate ```.hdf5``` files from the train/test sets.

## Training
To start training the network model:
- change directory to ```$DEEPSEGMENT_DIR```, and
- execute the command ```python train.py -c <path-to-ini>```

where ```<path-to-ini>``` points to the ```.ini``` file with required configuration parameters. See ```liver_coronal_unet.ini``` or ```abdotho_axial_unet.ini``` under ```$DEEPSEGMENT_DIR/projects/MSD_full/configs/``` for example configurations.

To reproduce our manuscript's results, execute the following command instead:

```python train.py -c <path-to-ini> \```\
```--loss <loss_function> \```\
```--train-weightage-mask <presence_mask_mode> \```\
```--checkpoint-path <unique_ckpt_dir>```

where:
- ```<path-to-ini>``` is any of the 12 ```.ini``` files under ```$DEEPSEGMENT_DIR/projects/MSD_full/configs/```
- ```<loss_function>``` is any of the following: ```xent```, ```xent+0.1*softdice```, ```xent+0.1*logdice```
- ```<presence_mask_mode>``` is any of the following: ```base```, ```or```, ```plus```
- ```<unique_ckpt_dir>``` is path to any directory (e.g. ```projects/MSD_full/log/liver_axial_xentsoftdice_plus```)

### configurations
```train.py``` loads configurations from both the command line arguments and the ```.ini``` config file, with command line arguments preceding over the config file. For a complete list of configuration parameters available, execute the command ```python train.py -h```

## Inference
Steps for label prediction is very similar to model training:
- change directory to ```$DEEPSEGMENT_DIR```, and
- execute the command ```python predict.py -c <path-to-ini> --checkpoint-path <ckpt_dir>```

This will load the trained model from ```<ckpt_dir>``` or, if not specified in the command line, ```path``` under the ```[checkpoint]``` section of the ```.ini``` file ```<path-to-ini>```

### generating predictions for NIfTI images
To generate predictions using NIfTI images directly (i.e. instead of generating the intermediate HDF5 files), execute the following command instead:

```python predict.py -c <path-to-ini> \```\
```--checkpoint-path <ckpt_dir> \```\
```--image-type nii \```\
```--image-paths <path-to-nii-input> \```\
```--predict-imgkey-feature . \```\
```--predict-output-path <path-to-nii-ouput>```

where:
- ```<path-to-nii-input>``` is path to directory containing the NIfTI feature images (e.g. ```projects/MSD_full/data/nii/Task03_Liver/image_test/```)
- ```<path-to-nii-ouput>``` is path to directory where the predicted NIfTI label images should be stored (e.g. ```projects/MSD_full/predictions/liver_axial_xentsoftdice_plus/```)

## Developer's notes

### adding new configuration parameter
New configuration parameter has to be added as property of the ```DeepSegmentConfig``` class, which is defined in ```$DEEPSEGMENT_DIR/utilities/config.py```. Then, define the property setter with optional checkings and type conversion. Make sure that it is also able to accept ```None``` as the input value, which is required by the initialization mechanism.

```load_configuration()```, defined also in ```$DEEPSEGMENT_DIR/utilities/config.py```, will then create an instance of the ```DeepSegmentConfig``` class and populate it with config parameters specified in a config ```.ini``` file as well as command line arguments. Any class property not specified in both the config file and command line will have the default value of ```None```.

#### config ```.ini``` file
Config ```.ini``` file is parsed with Python's ```configparser``` prior to being loaded into the ```DeepSegmentConfig``` instance. For each section and entry (```<key>: <value>```) listed in the ```.ini``` file, ```<value>``` will be stored in ```DeepSegmentConfig```'s property with the name ```<section>_<key>```.

#### command line argument
Command line arguments are parsed with Python's ```argparser```. New parameters can be defined and added to the parser under ```_create_argparser()```, defined also in ```$DEEPSEGMENT_DIR/utilities/config.py```. Values specified via the command line will overload values specified in the ```.ini``` file (if any).

Default value should not be specified while adding new parameter to the parser, since it will always overload values specified in the ```.ini``` file.
