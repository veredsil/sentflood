PyExample Program

Installation:
1. Retrieve the dataset: gsutil -m rsync -r gs://sen1floods11
2. cd sentfloods
3. Create venv: python -m venv venv
4. pip install -r ./requirements.txt

Execution:
1. cd pyexample 
2. python main.py


Step 1 - Basic Statistics
1. Number of images in each split and each region.
    step1_stats.py: step1_number_of_images()
2. Per-channel mean and standard deviation:
    step1_stats.py: 
        entire dataset: step1_stats.py: step1_mean_stdv_bands_allsplits()
        per split: step1_stats.py: step1_mean_stdv_bands_allsplits_persplit()
    
3. Probability of water (based on labels)
    a. Per image
    b. Per train/dev/test sets and for the held-out region (Bolivia)
    
    step1_stats.py: step1_water_probability_per_image_persplit()


Step 2 - Using NDWI to predict water
computes the per-pixel probability of water using the NDWI index (specifically using B03 and B08 bands).
and finds the optimal NDWI index threshold using MCC on the training set. Results are evaluated on the test and Bolivia splits.

    step2_calc_ndwi(): calculates ndwi for the train/valid and test/bolivia datasets
                        finds the optimal threshold using: optimal_ndwi_th_mcc()
                        MCC was chosen for better performance in imbalanced representation of the labels (as seen in Step 1)

    hist_ndwi_water_dry() - shows the distribution of the NDWI values for dry/water pixels. There is a large overlap, so will use MCC to find the optimal threshold.

    batch_visualize_ndwith_eval_withrgb() - visualizing random frames from the training set using the optimal threshold
                            blue - water, grey - dry, black - bad pixels
    

    The optimal threshold was found to be: 


Step 3 - Create a TFRecord file

Saving the original dataset (with the splits) into a TFRecord format, which is documented here:
https://www.tensorflow.org/tutorials/load_data/tfrecord 
The image bytes are be stored as a tf.train.BytesList feature, when the contents is an array of [H, W, C=13] uint16 bytes. The labels are stored in a format similar to the image, as an array of [H,W,C=1]

step3_tfrec.py: step3_tfrecoeds_for_dataset() - creates a tfrecoeds for train, valid, test, and bolivia splits
                step3_tfrecoeds_visual() - visualize the records


Please create a python script that reads the original files and creates a single tf.data.Example record for each input image.??
Optional: Add a colab to visualize the contents of the TFRecord file
