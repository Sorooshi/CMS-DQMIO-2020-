import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt


def parse_argument(args_):
    base_path_ = args_.Base_path
    number_of_run_ = args_.Number_of_number
    name_of_histogram_ = args_.Name_of_histogram
    return base_path_, number_of_run_, name_of_histogram_


def load_data_set(path_to_data_array, path_to_data_labels):

    data_array = np.loadtxt(path_to_data_array)
    data_labels = np.loadtxt(path_to_data_labels)
    good_data = data_array[np.where(data_labels == 1)]
    bad_data = data_array[np.where(data_labels == 0)]
    good_data = good_data[0]
    bad_data = bad_data[0]
    data_pd = pd.DataFrame(data=data_array)

    return data_array, data_labels, good_data, bad_data, data_pd


def plot_pca(data_array):
    return None


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--Base_path', type=str, default='/home/sshalileh/ml4dc/',
                        help='Enter run number')
    parser.add_argument('--Number_of_number', type=str, default='297050',
                        help='Enter run number')
    parser.add_argument('--Name_of_histogram', type=str, default='goodvtxNbr',
                        help='Name of the histogram for creating the Numpy array')

    args = parser.parse_args()

    base_path, number_of_run, name_of_histogram = parse_argument(args_=args)

    path_to_data_array = os.path.join(base_path,
                                      'matrices/' + number_of_run + "-" + name_of_histogram + ".npy")

    path_to_data_labels = os.path.join(base_path,
                                       'matrices/' + number_of_run + "-" + name_of_histogram + "-labels.npy")

    data_array, data_labels, good_data, bad_data, data_pd = load_data_set(path_to_data_array=path_to_data_array,
                                                                          path_to_data_labels=path_to_data_labels)



