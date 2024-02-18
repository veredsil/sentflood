import tensorflow as tf
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

output_path = 'output/step3/'
if not os.path.exists(poutput_path):
    os.makedirs(output_path)
if not os.path.exists(os.path.join(output_path, 'tfrecords')):
    os.makedirs(os.path.join(output_path, 'tfrecords'))

def load_s2filelist_df(splits):
    """
    load .tif files list into pandas dataframe for desired splits 
    listed in the corresponding csv files in: dataset/flood_handlabeled/'
    input: splits, e.g: ['train','test']
    output: pandas dataframe with columns 'region','imageId','sampleType', 'split'
    """
    data_root = 'dataset/flood_handlabeled/'
    df = []
    for split in splits:
        df_tmp = pd.read_csv(os.path.join(data_root, f'flood_{split}_data.csv'))
        df_tmp.columns = ['S1Hand','LabelHand']
        df_tmp[['region','imageId','sampleType']] = df_tmp['S1Hand'].str.split('_', expand=True)
        df_tmp['split'] = split
        df.append(df_tmp)

    df = pd.concat(df)
    return df

def s2data_to_imagebytes(imgPath):
    """ read S2 tif and conver to bytes """
    with rasterio.open(imgPath) as src:
        image_data = src.read()
        image_data = np.transpose(np.array(image_data), (1, 2, 0))  # Change order to put bands at the end
        image_data = image_data.astype(np.int16)
    return image_data.tobytes()

def serialize_S2img(filename):
    """ load S2 data and labels and create a tf.train record"""

    s2_path = 'dataset/S2Hand/'
    label_path = "dataset/LabelHand/"
    
    fnamesplit = filename.split('_')
    s2_imgpath = os.path.join(s2_path, f'{fnamesplit[0]}_{fnamesplit[1]}_S2Hand.tif')
    label_imgpath = os.path.join(label_path, f'{fnamesplit[0]}_{fnamesplit[1]}_LabelHand.tif' )
    
    image_bytes_data = s2data_to_imagebytes(s2_imgpath)
    image_bytes_mask = s2data_to_imagebytes(label_imgpath)
    
    feature = {
            "image": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes_data])),
            "label": tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes_mask]))
        }
    
    return tf.train.Example(features=tf.train.Features(feature=feature))

def parse_tfrecord_fn(example):
    """ parse tf.train record before decoding"""
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.string)}
    
    return tf.io.parse_single_example(example, feature_description)

def decode_image(image_raw):
    image = tf.io.decode_raw(image_raw, out_type='int16')
    image = tf.reshape(image, [512, 512, 13])
    return image

def decode_label(image_raw):
    image = tf.io.decode_raw(image_raw, out_type='int16')
    image = tf.reshape(image, [512, 512, 1])
    return image

def s2stack_to_rgb(img_tmp):
    """ create rgb image from 13 band array """
    img_rgb = np.dstack((img_tmp[:,:,3], img_tmp[:,:,2], img_tmp[:,:,1]))  
    return img_rgb

def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min())

def write_tfrecord(df, tfrecord_file_path):
    """ write tfrecord for all files in df , save to tfrecord_file_path"""
    with tf.io.TFRecordWriter(tfrecord_file_path) as writer:
        for file_path in df['LabelHand'].values:
            tf_example = serialize_S2img(file_path)
            serialized_example = tf_example.SerializeToString()
            writer.write(serialized_example)
        
    return

def mask_to_rgb(mask_bin):
    """ rgb from masked array so that: 
        blue: water
        grey: dry
        black: unused pixels
    """
    mask_rgb = np.zeros((*mask_bin.shape, 3), dtype=np.uint8)
    mask_rgb[mask_bin == 1] = [0, 0, 255]
    mask_rgb[mask_bin == 0] = [128, 128, 128]
    
    return mask_rgb

def visualize_parsed_record(parsed_record):
    """ visualize a tf.train record after parsing """

    img_tmp = decode_image(parsed_record['image'].numpy())
    label_tmp = decode_label(parsed_record['label'].numpy())
    label_tmp_rgb = mask_to_rgb(np.squeeze(label_tmp))
    rgb_image = s2stack_to_rgb(img_tmp)
    rgb_image = (rgb_image - tf.reduce_min(rgb_image)) / (tf.reduce_max(rgb_image) - tf.reduce_min(rgb_image))
        
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        
    axs[0].imshow(rgb_image)
    axs[0].set_title('RGB Image')
    axs[0].axis('off')

    axs[1].imshow(label_tmp_rgb)
    axs[1].set_title('Ground Truth')
    axs[1].axis('off')

    plt.show()

    return fig

def step3_tfrecoeds_for_dataset():
    """ save tf.record files for all splits"""

    splits = ['train','valid','test','bolivia']
    for split in splits:
        df_split = load_s2filelist_df([split])
        write_tfrecord(df_split, os.path.join(output_path, 'tfrecords', f'tfrecords.{split}'))
        print(f'tfrecords.{split} saved')

    return

def step3_tfrecoeds_visual():
    """ visualize some tf.records """
    splits = ['train','valid','test','bolivia']
    num_images = 3 # plot 3 images from sample
    for split in splits:
        tfrecord_file_path = f'tfrecords.{split}'
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file_path)
        parsed_dataset = raw_dataset.map(parse_tfrecord_fn)
        for ii in range(num_images):
            for parsed_record in parsed_dataset.take(ii):
                fig = visualize_parsed_record(parsed_record)
                fig.savefig(os.path.join(output_path, f'step3_tfrec_vis_{split}_{ii:02d}.png'), dpi=300)
    return

step3_tfrecoeds_for_dataset()
step3_tfrecoeds_visual()