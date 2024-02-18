# Step 1 - Basic Statistics
# Write a script that computes the basic statistics of the dataset
# Nmumber of images in each split and each region.
# Per-channel mean and standard deviation.
import os
import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

output_path = 'output/'

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

def get_s2_hand(filename):
    """ 
    loads S2 data files- 13 band tif and corresponding labeled mask
    input: file name 'REGION_IMGID_SPLIT.tif'
    output: data_s2 (512,512,13) , data_label(512, 512, 1) , np.array
    """
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

def step1_number_of_images():
    """
    calculate  number of images per split per region
    """
    # load all S2 datafiles, exclude bolivia
    df_allbol = load_s2filelist_df(splits=['train','valid','test','bolivia'])
    df_all = df_allbol[df_allbol['split'] != "bolivia"]

    # calc number of images per split per region
    df_stats = df_all.groupby(['split', 'region'])['imageId'].nunique().reset_index()
    df_order_freq = df_all.groupby('region')['imageId'].nunique().rename('total').reset_index().sort_values(by='total', ascending=False)
    df_stats_order = pd.merge(df_stats, df_order_freq[['region']], on='region', how='left')

    fig = plt.figure(figsize=(12, 7))
    sns.barplot(x='split', y='imageId', hue='region', data=df_stats_order, hue_order=df_order_freq['region'])
    plt.title('Number of images per split and region', fontsize=18)
    plt.xlabel('', fontsize=16)
    plt.ylabel('Number of images', fontsize=18)
    plt.legend(title='Region', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(fontsize=18)
    plt.tight_layout()
    fig.savefig(os.path.join(output_path,'step1_images_per_split_and_region.png'), dpi=240)
    plt.show()

    df_stats_order.to_csv(os.path.join(output_path,'step1_df_num_of_images_summary.csv'))

    return df_stats_order, fig

def stats_per_channel_image(label_path):
    """
    input: filename of LabelHand
    output: sum and sum of squeares per band over all valid pixels
    """
    data_s2, data_label = get_s2_hand(label_path)
    data_label_broad = data_label * np.ones([512, 512, 13])
    data_s2_masked = np.where(data_label_broad > -1, data_s2, np.nan)
    band_sums = np.nansum(data_s2_masked, axis=(0,1))
    band_sumsquare = np.nansum(data_s2_masked*data_s2_masked, axis=(0,1))
    pixel_count = np.count_nonzero(data_label > -1, axis=(0, 1))

    return band_sums, band_sumsquare, pixel_count
    
def step1_mean_stdv_bands_fromdf(df):
    """
    Calculate Per-channel mean and standard deviation per band for the entire dataset
    input: pandas dataframe containing all the filenames df['LabelHand']
    output: pandas dataframe with mean and standard deviation per band
    """
    nbands = 13
    pixel_count = 0
    sums = np.zeros([nbands])
    sums_square = np.zeros([nbands])

    for label_path in df['LabelHand'].values:
        band_sums, band_sumsquare, pixel_count = stats_per_channel_image(label_path)
        sums += band_sums
        sums_square += band_sumsquare
        pixel_count += pixel_count

    bands_mean = sums / pixel_count
    bands_stdv = np.sqrt((sums_square / pixel_count) - bands_mean**2)
    df_band_stats = pd.DataFrame()
    df_band_stats['band'] = range(1, nbands+1)
    df_band_stats['mean'] = bands_mean
    df_band_stats['std'] = bands_stdv

    return df_band_stats

def step1_mean_stdv_bands_allsplits():
    """
    Load all files from all splits and 
    Calculate Per-channel mean and standard deviation per band for the entire dataset
    input: pandas dataframe containing all the filenames df['LabelHand']
    output: pandas dataframe with mean and standard deviation per band
            the dataframe is saved to a .csv
    """
    
    df_allbol = load_s2filelist_df(splits=['train','valid','test','bolivia'])
    df_band_stats = step1_mean_stdv_bands_fromdf(df_allbol)
    df_band_stats.to_csv(os.path.join(output_path,'step1_df_band_stats_sum_all.csv'))

    print(df_band_stats)

    return

def step1_mean_stdv_bands_allsplits_v2():
    """
    Calculate Per Image Per-channel mean and standard deviation per band for the entire dataset
    input: none
    output: pandas dataframe with sum and square sum per band per image
            the dataframe is saved to a .csv
    """
    df_allbol = load_s2filelist_df(splits=['train','valid','test','bolivia'])
    stats_v2 = df_allbol['LabelHand'].apply(lambda x: stats_per_channel_image(x))
    df_band_stats = df_allbol.copy()
    sums_columns = [f'band{ii:02d}_sum' for ii in range(1, 14)]
    sumsquare_columns = [f'band{ii:02d}_sumsq' for ii in range(1, 14)]
    df_band_stats[sums_columns] = pd.DataFrame(stats_v2.apply(lambda x: x[0]).tolist(), index=df_band_stats.index)
    df_band_stats[sumsquare_columns] = pd.DataFrame(stats_v2.apply(lambda x: x[1]).tolist(), index=df_band_stats.index)
    df_band_stats['pixel_count'] = pd.DataFrame(stats_v2.apply(lambda x: x[2]).tolist(), index=df_band_stats.index)
    print(df_band_stats)
    df_band_stats.to_csv(os.path.join(output_path,'step1_df_band_stats_all_frames.csv'))

    return df_band_stats

def step1_mean_stdv_bands_allsplits_persplit():
    """
    Calculate Per Image Per-channel mean and standard deviation per band for the entire dataset
    input: none
    output: pandas dataframe with mean and std per band per split
            the dataframe for mean and std saved to a .csv
    """
    df_band_stats = step1_mean_stdv_bands_allsplits_v2()
    df_band_std_splits = []
    df_band_means_splits = []
    for ii in range(1,14):
        df_mean = df_band_stats.groupby('split').apply( lambda x: x[f'band{ii:02d}_sum'].sum() / (x['pixel_count'].sum()))
        df_std = df_band_stats.groupby('split').apply( lambda x: np.sqrt(x[f'band{ii:02d}_sumsq'].sum()/x['pixel_count'].sum() - x[f'band{ii:02d}_sum'].sum() / (x['pixel_count'].sum())))
        df_mean = df_mean.to_frame(name=f'band{ii:02d}')
        df_std = df_std.to_frame(name=f'band{ii:02d}')
        df_band_means_splits.append(df_mean.T)
        df_band_std_splits.append(df_std.T)

    df_band_std_splits = pd.concat(df_band_std_splits)
    df_band_means_splits = pd.concat(df_band_means_splits)

    df_band_means_splits.to_csv(os.path.join(output_path,'step1_df_band_means_splits.csv'))
    df_band_std_splits.to_csv(os.path.join(output_path,'step1_df_band_std_splits.csv'))

    print('Per channel mean per split:')
    print(df_band_means_splits)
    print('===========')
    print('Per channel std per split:')
    print(df_band_std_splits)

    return df_band_means_splits, df_band_std_splits



def water_probability_image(label_path):
    """ 
    load Sentinel-2 data (13 bands and labels) 
    calculate the number of water pixels based on handlabels
    """
    _ , data_label = get_s2_hand(label_path)
    pixels_water = np.sum(np.where(data_label == 1, 1, 0))
    pixels_dry = np.sum(np.where(data_label == 0, 1, 0))
    pixels_nan = np.sum(np.where(data_label == -1, 1, 0))

    return pixels_water, pixels_dry, pixels_nan

def step1_water_probability_per_image_persplit():
    """ 
    load Sentinel-2 dataset from all splits
    calculate the number of water pixels based on handlabels
    """
    df_allbol = load_s2filelist_df(splits=['train','valid','test','bolivia'])
    df_waterprob = df_allbol.copy()
    df_waterprob[['pixels_water', 'pixels_dry', 'pixels_nan']] = df_waterprob['LabelHand'].apply(lambda x: pd.Series(water_probability_image(x)))
    df_waterprob['prob_water'] = df_waterprob['pixels_water']/(df_waterprob['pixels_water']+df_waterprob['pixels_dry'])
    df_waterprob['prob_water'] = np.where((df_waterprob['pixels_water']+df_waterprob['pixels_dry']) == 0, 0, df_waterprob['prob_water'])\

    # plot hist of water probability in each frame
    fig = plt.figure(figsize=(10, 6))
    plt.hist(df_waterprob['prob_water'].dropna(), bins=50, color='blue', alpha=0.7)
    plt.title('Water probability per image', fontsize=18)
    plt.xlabel('Water probability (fraction of water pixels)', fontsize=16)
    plt.xticks(fontsize=16) 
    plt.ylabel('# of images', fontsize=16)
    plt.grid(True)
    fig.savefig(os.path.join(output_path,'step1_water_probability_per_image.png'), dpi=300)
    plt.show()
    print('Distribution of probability of water per image was saved to: step1_water_probability_per_image.png')
    
    # Probability of water (based on labels)
    df_waterprob_split = df_waterprob.groupby('split').apply(lambda x: \
         0 if (x['pixels_water'].sum() + x['pixels_dry'].sum()) == 0 else x['pixels_water'].sum() / (x['pixels_water'].sum() + x['pixels_dry'].sum()))
    print('====')
    print('Probability of water per split:')
    print(df_waterprob_split)
    df_waterprob_split.to_csv(os.path.join(output_path,'df_waterprob_split.csv'))
    df_waterprob.to_csv(os.path.join(output_path,'df_waterprob.csv'))

    return df_waterprob, df_waterprob_split


# call routine:
print('==============================')
print('Step1 - Number of images in each split and each region:')
step1_number_of_images()


print('==============================')
print('Step1 - Per-channel mean and standard deviation:')
df_band_means_splits, df_band_std_splits = step1_mean_stdv_bands_allsplits_persplit()

print('==============================')
print('Step1 - Probability of water:')
df_waterprob, df_waterprob_split = step1_water_probability_per_image_persplit()