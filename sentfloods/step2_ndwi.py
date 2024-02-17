from PIL import Image
import pandas as pd
import rasterio
from rasterio.plot import reshape_as_image
import os
from PIL import Image
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import matthews_corrcoef

output_path = 'output/'

def load_s2filelist_df(splits):
    # load S2 file list into pandas dataframe for desired splits
    # input: splits (list)
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

def get_s2_hand(filename):
    # loads S2 data files- 13 band tif and corresponding labeled mask
    # input: file name 'REGION_IMGID_SPLIT.tif'
    # output: data_s2 (512,512,13) , data_label(512, 512, 1) , np.array
    path_s2 = 'dataset/S2Hand/'
    path_label = "dataset/LabelHand/"
    fnamesplit = filename.split('_')
    s2_imgpath = os.path.join(path_s2, f'{fnamesplit[0]}_{fnamesplit[1]}_S2Hand.tif' )
    label_imgpath = os.path.join(path_label, f'{fnamesplit[0]}_{fnamesplit[1]}_LabelHand.tif' )
    with rasterio.open(s2_imgpath) as src:
        data_s2 = src.read().astype(float).T

    with rasterio.open(label_imgpath) as src:
        data_label = src.read().astype(float).T

    return data_s2, data_label


def calc_ndwi(nir, green):
    ndwi = np.where((nir + green) == 0, 0, (nir - green) / (green + nir))
    return ndwi

def s2stack_to_rgb(img_tmp):
    img_rgb = np.dstack((img_tmp[:,:,3], img_tmp[:,:,2], img_tmp[:,:,1]))
    return img_rgb / 10000

def calc_mndwi_from_df(df):
    ndwis = []
    mndwis = []
    labels = []
    rgbs = []

    for filename in df['LabelHand'].values:   
        data_s2, data_label = get_s2_hand(filename)
        ndwi = calc_ndwi(data_s2[:,:,3]/10000, data_s2[:,:,8]/10000)
        mndwi = calc_ndwi(data_s2[:,:,3]/10000, data_s2[:,:,11]/10000)
        img_rgb = s2stack_to_rgb(data_s2)
        rgbs.append(img_rgb)
        ndwis.append(ndwi)
        mndwis.append(mndwi)
        labels.append(data_label)
    
    ndwis = np.stack(ndwis).transpose((1,2,0))
    mndwis = np.stack(mndwis).transpose((1,2,0))
    labels = np.stack(labels).transpose((1,2,3,0))
    rgbs = np.stack(rgbs).transpose((1,2,3,0))

    return ndwis, labels, rgbs, mndwis


def hist_ndwi_water_dry(ndwi_arr, label_arr):
    ndwi_filt = ndwi_arr.flatten()
    label_filt = label_arr.flatten()
    ndwi_filt = ndwi_filt[label_filt>-1]
    label_filt = label_filt[label_filt>-1]

    water_ndwi = ndwi_filt[label_filt == 1]
    non_water_ndwi = ndwi_filt[label_filt == 0]

    fig = plt.figure(figsize=(10, 6))

    plt.hist(water_ndwi.flatten(), bins=50, alpha=0.5, label='Water NDWI', color='blue', density=True)
    plt.hist(non_water_ndwi.flatten(), bins=50, alpha=0.5, label='Non-Water NDWI', color='red', density=True)
    plt.title('NDWI distribution for water and dry pixels')
    plt.xlabel('NDWI')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    return fig

def optimal_ndwi_th_mcc(label_all, ndwi_all):

    ndwi_filt = ndwi_all.flatten()
    label_filt = label_all.flatten()
    ndwi_filt = ndwi_filt[label_filt>-1]
    label_filt = label_filt[label_filt>-1]

    # Initialize an array to store MCC for each threshold
    mcc_scores = []
    thresholds = np.linspace(-.3, .1, 15)

    # Calculate MCC for each threshold
    for threshold in thresholds:
        predictions = ndwi_filt > threshold
        mcc = matthews_corrcoef(label_filt, predictions)
        mcc_scores.append(mcc)

    # Find the threshold with the highest MCC
    max_mcc_index = np.argmax(mcc_scores)
    optimal_threshold = thresholds[max_mcc_index]
    print(f"Optimal NDWI Threshold: {optimal_threshold} with MCC: {mcc_scores[max_mcc_index]}")

    return mcc_scores, thresholds, optimal_threshold

def mask_to_rgb(mask_bin):
    
    mask_rgb = np.zeros((*mask_bin.shape, 3), dtype=np.uint8)
    mask_rgb[mask_bin == 1] = [0, 0, 255]
    mask_rgb[mask_bin == 0] = [128, 128, 128]
    
    return mask_rgb

def visualize_ndwi_eval_withrgb(ndwi, label, ndwi_th, img_rgb):
    
    fig, axs = plt.subplots(1, 3, figsize = (12, 6))
    axs = axs.flatten()
    
    ndwi_optimal_mask = np.where(ndwi>ndwi_th, 1, 0)
    ndwi_optimal_mask = np.where(label>-1, ndwi_optimal_mask, -1)
    
    ndwi_optimal_mask_rgb = mask_to_rgb(ndwi_optimal_mask)
    label_rgb = mask_to_rgb(label)

    cax0 = axs[0].imshow(img_rgb)
    axs[0].set_title('RGB')

    cax3 = axs[1].imshow(label_rgb)
    axs[1].set_title('Ground Truth')

    cax2 = axs[2].imshow(ndwi_optimal_mask_rgb)
    axs[2].set_title('NDWI mask')
    
    plt.tight_layout()

    return fig

def normalize_band(band):
    return (band - band.min()) / (band.max() - band.min())

def batch_visualize_ndwith_eval_withrgb(df, ndwi_th):
    
    inds_plot = np.random.choice(range(len(df)), 4, replace=False)
    
    for ii in inds_plot:
        filename = df['LabelHand'].iloc[ii] 
        data_s2, data_label = get_s2_hand(filename)
        ndwi = calc_ndwi(data_s2[:,:,3]/10000, data_s2[:,:,8]/10000)
        img_rgb = s2stack_to_rgb(data_s2)
        red_normalized = normalize_band(img_rgb[:,:,0])
        green_normalized = normalize_band(img_rgb[:,:,1])
        blue_normalized = normalize_band(img_rgb[:,:,2])
        rgb_image = np.stack([red_normalized, green_normalized, blue_normalized], axis=-1)

        fig = visualize_ndwi_eval_withrgb(ndwi, \
                                        data_label[:,:,0], ndwi_th, rgb_image)

        fig.savefig(os.path.join(output_path, f'step2_ndwi_threshold_vis_{filename[:-10]}.png'), dpi=300)

    return fig

def step2_calc_ndwi():

    df_trainvalid = load_s2filelist_df(['train','valid'])
    df_testbol = load_s2filelist_df(['test','bolivia'])

    ndwi_train, label_train, rgb_train, mndwi_train = calc_mndwi_from_df(df_trainvalid)
    ndwi_test, label_test, rgb_test, mndwi_test = calc_mndwi_from_df(df_testbol)

    fig = hist_ndwi_water_dry(ndwi_train, label_train)
    fig.savefig(os.path.join(output_path,'step2_ndwi_hist_water_dry_trainvalid.png'), dpi=300)
    print('Distribution of NDWI for water and dry pixels for train/valid splits saved to: step2_ndwi_hist_water_dry_trainvalid.png')
 
    fig = hist_ndwi_water_dry(ndwi_test, label_test)
    fig.savefig(os.path.join(output_path,'step2_ndwi_hist_water_dry_testbol.png'), dpi=300)
    print('Distribution of NDWI for water and dry pixels for test/valboliviaid splits saved to: step2_ndwi_hist_water_dry_testbol.png')
 
    mcc_scores, thresholds, optimal_threshold = optimal_ndwi_th_mcc(label_train, ndwi_train)

    max_mcc_index = np.argmax(mcc_scores)
    optimal_threshold = thresholds[max_mcc_index]
    print(f"Optimal NDWI Threshold: {optimal_threshold} with MCC: {mcc_scores[max_mcc_index]}")

    # Plot MCC vs. threshold
    fig = plt.figure(figsize=(10, 6))
    plt.plot(thresholds, mcc_scores, label='MCC Score')
    plt.plot(optimal_threshold, mcc_scores[max_mcc_index], 'ro')  # Mark the optimal threshold
    plt.title('MCC vs. NDWI Threshold')
    plt.xlabel('NDWI Threshold')
    plt.ylabel('MCC Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    fig.savefig(os.path.join(output_path,'step2_ndwi_hist_water_dry_testbol.png'), dpi=300)

    batch_visualize_ndwith_eval_withrgb(df_testbol, optimal_threshold)

    return


step2_calc_ndwi()