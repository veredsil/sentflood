# Flood detection using Sentinel-2 recoreds from the Sen1Floods11 dataset

## Description

The Sen1Floods11 dataset (dataset, paper) is a georeferenced dataset to train and test deep learning flood algorithms for Sentinel-1. Here we use the human-annotated (hand labeled) subset of Sentinel-2 images. The data is described in Section 2.2 (Flood Event Data) of the paper:
https://openaccess.thecvf.com/content_CVPRW_2020/papers/w11/Bonafilia_Sen1Floods11_A_Georeferenced_Dataset_to_Train_and_Test_Deep_Learning_CVPRW_2020_paper.pdf


In order to download the dataset you’ll need to install google cloud command line interface and run “gcloud auth login” to set up authentication, before you can use the gsutil command.

## Usage

1. Retrieve the dataset: 

gsutil -m rsync -r gs://sen1floods11/v1.1/splits/flood_handlabeled/
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand
gsutil -m rsync -r gs://sen1floods11/v1.1/data/flood_events/HandLabeled/S2Hand

the final file structure should be:
    dataset/flood_handlabeled
    dataset/S2Hand
    dataset/LabelHand

2. cd sentfloods
3. Create venv: python -m venv venv
4. pip install -r ./requirements.txt

## Features

### Step 1 - Basic Statistics
1. Number of images in each split and each region.
    step1_stats.py: step1_number_of_images()

![][https://github.com/veredsil/sentflood/blob/fae05c6cd5533795ea801d5c53d016d32b8ce59e/sentfloods/output/step1_water_probability_per_image.png]

2. Per-channel mean and standard deviation:
    step1_stats.py: 
        entire dataset: step1_stats.py: step1_mean_stdv_bands_allsplits()
        per split: step1_stats.py: step1_mean_stdv_bands_allsplits_persplit()
    
3. Probability of water (based on labels)
    a. Per image
    b. Per train/dev/test sets and for the held-out region (Bolivia)
    
    step1_stats.py: step1_water_probability_per_image_persplit()

###  Step 2 - Using NDWI to predict water
Computes the per-pixel probability of water using the NDWI index (specifically using B03 and B08 bands).
and finds the optimal NDWI index threshold using MCC on the training set. Results are evaluated on the test and Bolivia splits.

    step2_calc_ndwi(): calculates ndwi for the train/valid and test/bolivia datasets
                        finds the optimal threshold using: optimal_ndwi_th_mcc()
                        MCC was chosen for better performance in imbalanced representation of the labels (as seen in Step 1) 
                        the range of thresholds was chosen according to the labeled distributions

    hist_ndwi_water_dry() - shows the distribution of the NDWI values for dry/water pixels. There is a large    overlap, so will use MCC to find the optimal threshold. 

    batch_visualize_ndwith_eval_withrgb() - visualizing random frames from the training set using the optimal threshold
    blue - water, grey - dry, black - bad pixels

    Optimal NDWI Threshold: -0.0428 with MCC: 0.755


###  Step 3 - Create a TFRecord file

Saving the original dataset (with the splits) into a TFRecord format, which is documented here:
https://www.tensorflow.org/tutorials/load_data/tfrecord 
The image bytes are be stored as a tf.train.BytesList feature, when the contents is an array of [H, W, C=13] uint16 bytes. The labels are stored in a format similar to the image, as an array of [H,W,C=1]

step3_tfrec.py: step3_tfrecoeds_for_dataset() - creates a tfrecoeds for train, valid, test, and bolivia splits
                step3_tfrecoeds_visual() - visualize the records



