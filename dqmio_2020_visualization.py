import os
import pickle
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
import matplotlib.pyplot as plt
import data_normalization as dn
from matplotlib import colors as mcolors

HISTOGRAM_NAMES = [
    'goodvtxNbr', 'adc_PXLayer_1', 'adc_PXLayer_2', 'adc_PXLayer_3',
    'adc_PXLayer_4', 'adc_PXDisk_+1', 'adc_PXDisk_+2', 'adc_PXDisk_+3',
    'adc_PXDisk_-1', 'adc_PXDisk_-2', 'adc_PXDisk_-3',
     'num_clusters_ontrack_PXBarrel', 'num_clusters_ontrack_PXForward',
     'chargeInner_PXLayer_1', 'chargeInner_PXLayer_2',
     'chargeInner_PXLayer_3', 'chargeInner_PXLayer_4',
     'chargeOuter_PXLayer_1', 'chargeOuter_PXLayer_2',
    'chargeOuter_PXLayer_3', 'chargeOuter_PXLayer_4', 'size_PXLayer_1',
    'size_PXLayer_2', 'size_PXLayer_3', 'size_PXLayer_4',
    'charge_PXDisk_+1', 'charge_PXDisk_+2', 'charge_PXDisk_+3',
    'charge_PXDisk_-1', 'charge_PXDisk_-2', 'charge_PXDisk_-3',
    'size_PXDisk_+1', 'size_PXDisk_+2', 'size_PXDisk_+3',
    'size_PXDisk_-1', 'size_PXDisk_-2', 'size_PXDisk_-3',
    'MainDiagonal Position', 'NumberOfClustersInPixel',
    'NumberOfClustersInStrip', 'NormalizedHitResiduals_TEC__wheel__1',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__1',
    'NormalizedHitResiduals_TEC__wheel__2',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__2',
    'NormalizedHitResiduals_TEC__wheel__3',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__3',
    'NormalizedHitResiduals_TEC__wheel__4',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__4',
    'NormalizedHitResiduals_TEC__wheel__5',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__5',
    'NormalizedHitResiduals_TEC__wheel__6',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__6',
    'NormalizedHitResiduals_TEC__wheel__7',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__7',
    'NormalizedHitResiduals_TEC__wheel__8',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__8',
    'NormalizedHitResiduals_TEC__wheel__9',
    'Summary_ClusterStoNCorr__OnTrack__TEC__MINUS__wheel__9',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__1',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__2',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__3',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__4',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__5',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__6',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__7',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__8',
    'Summary_ClusterStoNCorr__OnTrack__TEC__PLUS__wheel__9',
    'NormalizedHitResiduals_TIB__Layer__1',
    'Summary_ClusterStoNCorr__OnTrack__TIB__layer__1',
    'NormalizedHitResiduals_TIB__Layer__2',
    'Summary_ClusterStoNCorr__OnTrack__TIB__layer__2',
    'NormalizedHitResiduals_TIB__Layer__3',
    'Summary_ClusterStoNCorr__OnTrack__TIB__layer__3',
    'NormalizedHitResiduals_TIB__Layer__4',
    'Summary_ClusterStoNCorr__OnTrack__TIB__layer__4',
    'NormalizedHitResiduals_TID__wheel__1',
    'Summary_ClusterStoNCorr__OnTrack__TID__MINUS__wheel__1',
    'NormalizedHitResiduals_TID__wheel__2',
    'Summary_ClusterStoNCorr__OnTrack__TID__MINUS__wheel__2',
    'NormalizedHitResiduals_TID__wheel__3',
    'Summary_ClusterStoNCorr__OnTrack__TID__MINUS__wheel__3',
    'Summary_ClusterStoNCorr__OnTrack__TID__PLUS__wheel__1',
    'Summary_ClusterStoNCorr__OnTrack__TID__PLUS__wheel__2',
    'Summary_ClusterStoNCorr__OnTrack__TID__PLUS__wheel__3',
    'NormalizedHitResiduals_TOB__Layer__1',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__1',
    'NormalizedHitResiduals_TOB__Layer__2',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__2',
    'NormalizedHitResiduals_TOB__Layer__3',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__3',
    'NormalizedHitResiduals_TOB__Layer__4',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__4',
    'NormalizedHitResiduals_TOB__Layer__5',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__5',
    'NormalizedHitResiduals_TOB__Layer__6',
    'Summary_ClusterStoNCorr__OnTrack__TOB__layer__6',
    'Chi2oNDF_lumiFlag_GenTk',
    'NumberOfRecHitsPerTrack_lumiFlag_GenTk',
    'NumberOfTracks_lumiFlag_GenTk',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__1',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__2',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__3',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__4',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__5',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__6',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__7',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__8',
    'Summary_TotalNumberOfDigis__TEC__MINUS__wheel__9',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__1',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__2',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__3',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__4',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__5',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__6',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__7',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__8',
    'Summary_TotalNumberOfDigis__TEC__PLUS__wheel__9',
    'Summary_TotalNumberOfDigis__TIB__layer__1',
    'Summary_TotalNumberOfDigis__TIB__layer__2',
    'Summary_TotalNumberOfDigis__TIB__layer__3',
    'Summary_TotalNumberOfDigis__TIB__layer__4',
    'Summary_TotalNumberOfDigis__TID__MINUS__wheel__1',
    'Summary_TotalNumberOfDigis__TID__MINUS__wheel__2',
    'Summary_TotalNumberOfDigis__TID__MINUS__wheel__3',
    'Summary_TotalNumberOfDigis__TID__PLUS__wheel__1',
    'Summary_TotalNumberOfDigis__TID__PLUS__wheel__2',
    'Summary_TotalNumberOfDigis__TID__PLUS__wheel__3',
    'Summary_TotalNumberOfDigis__TOB__layer__1',
    'Summary_TotalNumberOfDigis__TOB__layer__2',
    'Summary_TotalNumberOfDigis__TOB__layer__3',
    'Summary_TotalNumberOfDigis__TOB__layer__4',
    'Summary_TotalNumberOfDigis__TOB__layer__5',
    'Summary_TotalNumberOfDigis__TOB__layer__6']

plots_colors = ['blue', 'orange', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

summarization_colors = ['k', 'g']


line_styles = [(0, ()), (0, (1, 3)), (0, (3, 2))]


def parse_arguments(arguments):
    path_ = arguments.Path
    return path_


def load_process_a_df(path):

    """
     loads a Pandas DataFrame and sorts in ascending order regarding Run number and Lumisection number.
     Also converts all the strings are converted to integers.  
    :param path: path to load a Pandas DataFrame.
    :return: a preprocessed Pandas DataFrame
    """

    df = pd.read_csv(path)

    df = df.loc[(df['metype'] <= 5), ['fromrun', 'fromlumi', 'hname', 'histo',
                                      'entries', 'Xmax', 'Xmin', 'Xbins']]

    # covert the histograms from string into a list of int
    df['histo'] = df['histo'].apply(literal_eval)

    # assigning row indices w.r.t the two columns "fromrun" and "fromlumi"
    df.set_index(['fromrun', 'fromlumi'], inplace=True, drop=False)
    df.sort_index(axis=0, inplace=True)
    return df


def convert_df_to_np_array_for_per_histogram(df, name_of_histo):

    """
    loads a preprocessed DataFrame and returns:
      1) a dict of numpy array of corresponding histograms which is determined by the argument "name_of_histo";
      keys represent run number.
      2) a dict of dict; dict of statistics of each run. keys of the first dict represents the run number
      while the keys of the second dict represent the following basic statics (per each run) including:
        2-1) number of lumisections
        2-2) number of histograms
        2-3) average of length of histograms 
        2-4) standard deviation of length of histograms (should be zero always)
        2-5) average of histograms
        2-6) standard deviation of histograms
         
    :param df: a Pandas preprocessed DataFrame
    :param name_of_histo: name of histogram which one decides to use for further analysis
    :return: a dict of inconsistencies, a dict of histograms per each run with keys representing the run-number, a dict
    of some basic statistics.
    """

    inconsistencies = {}
    histograms = {}
    stats = {}

    for run in df['fromrun'].unique():
        print("run:", run)
        inconsistencies[run] = []
        n_histo, histos, histos_len = [], [], []
        stats[run] = {}
        for ls in df['fromlumi'][run]:
            try:
                n_histo.append(ls)  # number of iterators
                a_histo = df.loc[df['hname'] == name_of_histo]['histo'][run][ls]
                histos.append(a_histo)
                histos_len.append(len(a_histo))
            except (ValueError, KeyError, TypeError):
                inconsistencies[run].append((run, ls, name_of_histo))

        histos = np.asarray(histos)
        histos_len = np.asarray(histos_len)
        histo_names = df[(df['fromrun'] == run)]['hname'].unique()  # i.e 'hname'
        histograms[run] = histos

        stats[run]['ls'] = set(n_histo)
        stats[run]['n_ls'] = len(set(n_histo))
        stats[run]['histo_names'] = histo_names
        stats[run]['n_histo'] = len(histo_names)
        stats[run]['n_iters'] = len(n_histo)
        stats[run]['ave_len_histo'] = np.mean(histos_len, axis=0)
        stats[run]['std_len_histo'] = np.std(histos_len, axis=0)
        stats[run]['ave_histo'] = np.mean(histos, axis=0)
        stats[run]['std_histo'] = np.std(histos, axis=0)

    return histograms, stats, inconsistencies


def plot_per_histograms(histograms, name_of_histo, data_name,
                        path_to_store='/home/sshalileh/ml4dc/figs-2020'):

    runs_containing_constant_features = []

    for run, v in histograms.items():
        print("plotting " + name_of_histo + " in run: " + run)
        x_min, x_max = 0, 80000
        nbins = v.shape[1]
        fig, axes = plt.subplots(ncols=2, figsize=(10.5, 6.5))
        ax = axes[0]
        i = 0
        for row in range(v.shape[0]):
            a_hist = v[row]
            x = np.linspace(x_min, x_max, nbins)
            ax.step(x, a_hist, color=plots_colors[i % 9], alpha=0.9, )
            ax.set_xlabel('bins range', fontsize=10)
            ax.set_ylabel('entries', fontsize=10)
            ax.set_xlim([x_min, x_max])
            # ax.set_ylim([0.0, 1.05])
            ax.tick_params(axis='x', labelsize=10)
            ax.tick_params(axis='N', labelsize=10)
            hname = name_of_histo.split("_")
            hname = " ".join(hname)
            ax.set_title("run:" + str(run) + " " + hname, fontsize=12)
            i += 1
        ax = axes[1]
        x = np.linspace(x_min, x_max, nbins)
        hname = name_of_histo.split("_")
        hname = " ".join(hname)
        ax.step(x, np.mean(v, axis=0), color=summarization_colors[0],
                where='mid', label='All Histo. Ave.', alpha=0.9, )
        ax.step(x, np.std(v, axis=0), color=summarization_colors[1],
                where='mid', label='All Histo. Std.', alpha=0.9, )
        ax.set_xlabel('bins range', fontsize=10)
        ax.set_ylabel('entries', fontsize=10)
        ax.set_xlim([x_min, x_max])
        # ax.set_ylim([0.0, 1.05])
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='N', labelsize=10)
        ax.set_title(" run:" + str(run) + " " + hname, fontsize=12)
        x_cnst_free, x_rel_cntr, x_zsc, x_zsc_rel_cntr, x_rng, x_rng_rel_cntr, cnst_features = dn.preprocess_Y(
            Yin=v, nscf={})
        if x_cnst_free.shape[0] == 0:
            runs_containing_constant_features.append(run)
        plt.legend(loc='best')
        # plt.show()

        if not os.path.exists(path_to_store):
            os.mkdir(path_to_store)

        dir_path = os.path.join(path_to_store, data_name)
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        fig.savefig(os.path.join(dir_path, str(run) + '-' + hname + '.png'))

    return runs_containing_constant_features


if __name__ == '__main__':

    path_to_store_reviews = "/home/sshalileh/ml4dc/review-ds2020"

    parser = argparse.ArgumentParser()

    parser.add_argument('--Path', type=str, default='/home/fratnikov/cms/ml4dc/ML_2020/UL2017_Data',
                        help='Path to load a Pandas DataFrame')

    args = parser.parse_args()
    path_of_a_df = parse_arguments(arguments=args)

    print("path to data set:", path_of_a_df)

    # load and process a DataFrame:
    df_processed = load_process_a_df(path=path_of_a_df)

    histograms_total = {}
    stats_total = {}
    inconsistencies_total = {}
    runs_containing_constant_features_total = {}

    for name_of_histo in HISTOGRAM_NAMES:

        print("name of histpgram:", name_of_histo)

        histograms, stats, inconsistencies = convert_df_to_np_array_for_per_histogram(df=df_processed,
                                                                                      name_of_histo=name_of_histo
                                                                                      )

        tmp = path_of_a_df.split("/")
        tmp = tmp[-1]
        tmp = tmp.split("_")
        data_name = tmp[0] + "-" + tmp[-2] + "-" + tmp[-1].split(".")[0]

        runs_containing_constant_features = plot_per_histograms(histograms=histograms,
                            name_of_histo=name_of_histo,
                            data_name=data_name)

        histograms_total[name_of_histo] = histograms
        stats_total[name_of_histo] = stats
        inconsistencies_total[name_of_histo] = inconsistencies
        runs_containing_constant_features_total[name_of_histo] = runs_containing_constant_features

    if not os.path.exists(path_to_store_reviews):
        os.mkdir(path_to_store_reviews)

    tmp = path_of_a_df.split("/")
    tmp = tmp[-1]
    tmp = tmp.split("_")
    data_name_total = tmp[0] + "-" + tmp[1]

    with open(os.path.join(path_to_store_reviews,
                           data_name_total + 'stats_total.pickle'), 'wb') as fp:
        pickle.dump(stats_total, fp)

    with open(os.path.join(path_to_store_reviews,
                           data_name_total + 'histogram_total.pickle'), 'wb') as fp:
        pickle.dump(histograms_total, fp)

    with open(os.path.join(path_to_store_reviews,
                           data_name_total + 'inconsistencies_total.pickle'), 'wb') as fp:
        pickle.dump(inconsistencies_total, fp)

    with open(os.path.join(path_to_store_reviews,
                           data_name_total + 'run_with_constants_features_total.pickle'), 'wb') as fp:
        pickle.dump(runs_containing_constant_features_total, fp)
