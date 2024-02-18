import pandas as pd
import rasterio
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.metrics import f1_score, matthews_corrcoef, jaccard_score

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

def calc_ndwi(nir, green):
    """ 
        Calculate NDWI for Sentinel2 using band B03 for green, and band B08 for NIR 
        input: nir/green np.array[512, 512]
    """
    ndwi = np.where((nir + green) == 0, 0, (nir - green) / (green + nir))
    return ndwi

def s2stack_to_rgb(img_tmp):
    """ create rgb from Sentinel2 13 band array """
    img_rgb = np.dstack((img_tmp[:,:,3], img_tmp[:,:,2], img_tmp[:,:,1]))
    return img_rgb / 10000

def calc_mndwi_from_df(df):
    """ calculate ndwi from Sentinel2 13 band array for all files within df['LabelHand'] """
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
    """ 
    Plot histograms of NDWI values for water and dry pixels, to explore possible thresholds for NDWI
    """
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
    plt.grid()
    plt.tight_layout()

    return fig

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

        fig.savefig(os.path.join(output_path, f'step2_ndwi_threshold_vis_{filename[:-13]}.png'), dpi=300)

    return fig

def optimal_ndwi_th_mcc(label_all, ndwi_all):
    """
    Find optimal NDWI threshold my maximazing MCC score on the training set
    """

    # Remove pixels with label = -1
    ndwi_filt = ndwi_all.flatten()
    label_filt = label_all.flatten()
    ndwi_filt = ndwi_filt[label_filt>-1]
    label_filt = label_filt[label_filt>-1]

    mcc_scores = []
    f1_scores = []
    precisions = []
    recalls = []
    ious = []
    thresholds = np.linspace(-.25, .1, 20) # range was chosen according to the labeled distributions

    # Calculate MCC for each threshold
    for threshold in thresholds:
        predictions = ndwi_filt > threshold
        mcc = matthews_corrcoef(label_filt, predictions)
        mcc_scores.append(mcc)
        iou_score = jaccard_score(label_filt, predictions, average='binary')
        ious.append(iou_score)
        precision, recall, f1, _ = precision_recall_fscore_support(label_filt, predictions, average='binary')
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    # Find the threshold with the highest MCC
    max_mcc_index = np.argmax(mcc_scores)
    optimal_threshold = thresholds[max_mcc_index]
    print(f"Optimal NDWI Threshold: {optimal_threshold} with MCC: {mcc_scores[max_mcc_index]}")

    return thresholds, optimal_threshold, mcc_scores, f1_scores, precisions, recalls, ious

def visualize_metrics(thresholds, optimal_threshold, mcc_scores, f1_scores, precisions, recalls, ious):
    
    # plot F1 vs th
    fig = plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, label='Precision', marker='o')
    plt.plot(thresholds, recalls, label='Recall', marker='s')
    plt.plot(thresholds, f1_scores, label='F1 Score', marker='^')
    plt.plot(thresholds, mcc_scores, label='MCC Score', marker='D')
    plt.plot(thresholds, ious, label='IoU Score', marker='s')
    plt.legend()
    plt.plot(optimal_threshold, mcc_scores[max_mcc_index], 'ko')
    plt.plot(thresholds[np.argmax(f1_scores)], f1_scores[np.argmax(f1_scores)], 'ko')
    plt.plot(thresholds[np.argmax(ious)], ious[np.argmax(ious)], 'ko')

    plt.xlabel('NDWI Threshold')
    plt.ylabel('Score')
    plt.title('Validation Metrics vs. NDWI Threshold')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return fig

def evaluate_optimal_th_on_test(optimal_th, splits=['test','bolivia']):

    df_testbol = load_s2filelist_df(splits)
    ndwi_test, label_test, _, _ = calc_mndwi_from_df(df_testbol)
    label_test_filt = label_test[:,:,0]
    predictions_test = (ndwi_test > optimal_th)
    predictions_test = predictions_test[label_test_filt>-1]
    label_test_filt = label_test_filt[label_test_filt>-1]
    conf_matrix_test = confusion_matrix(label_test_filt, predictions_test)
    print("Test Confusion Matrix:\n", conf_matrix_test)

    mcc_test = matthews_corrcoef(label_test_filt, predictions_test)
    precision_test, recall_test, f1_test, _ = precision_recall_fscore_support(label_test_filt, predictions_test, average='binary')
    print(f"Test F1 Score: {f1_test:.3f}")
    print(f"Test MCC Score: {mcc_test:.3f}")
    
    return

def step2_calc_ndwi():
    """
    main function of step 2, calculates:
    1. NDWI for train/valid and test/bolivia splits
    2. Inspect distribution of NDWI values for water/dry pixels
    3. Find optimal threshold by maximizing performance metrics 
        over a relevant range of threshold values.
    4. Visualize the mask calculated from NDWI to the ground truth
    5. Evaluate the performance of the threshold on the testing splits

    """
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
 
    thresholds, optimal_threshold, mcc_scores, f1_scores, precisions, recalls, ious = optimal_ndwi_th_mcc(label_train, ndwi_train)

    df_metrics = pd.DataFrame({
        'Threshold': thresholds,
        'MCC Score': mcc_scores,
        'F1 Score': f1_scores,
        'Precision': precisions,
        'Recall': recalls, 
        'IoU': ious
    })

    # Display the DataFrame to ensure it looks correct
    print(df_metrics)

    df_metrics.to_csv(os.path.join(output_path,'optimal_ndwi_thresholds_trainvalid.csv'), index=False)
    max_mcc_index = np.argmax(mcc_scores)
    optimal_threshold = thresholds[max_mcc_index]
    print(f"Optimal NDWI Threshold: {optimal_threshold:0.3f} with MCC: {mcc_scores[max_mcc_index]:0.3f}")
    print(f"Optimal NDWI Threshold: {thresholds[f1_scores==np.max(f1_scores)][0]:.3f} with F1: {np.max(f1_scores):0.3f}")

    fig = visualize_metrics(thresholds, optimal_threshold, mcc_scores, f1_scores, precisions, recalls, ious)
    fig.savefig(os.path.join(output_path,'step2_ndwi_mcc_vs_th_train.png'), dpi=300)

    batch_visualize_ndwith_eval_withrgb(df_testbol, optimal_threshold)
    evaluate_optimal_th_on_test(optimal_threshold, splits=['test','bolivia'])

    return


step2_calc_ndwi()