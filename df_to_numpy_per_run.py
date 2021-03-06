import os
import glob
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from ast import literal_eval
from sklearn.model_selection import train_test_split


def parse_argument(args):
    run = args.Selected_run
    run = [int(i) for i in run.split(", ")]
    name_of_histo = args.Name_of_histo
    name_of_histo = name_of_histo.split(", ")
    return run, name_of_histo


def load_process_a_df(path):

    """
     loads a Pandas DataFrame and sorts in ascending order w.r.t the run number and the Lumisection number.
        Also converts all the strings to integers.
    :param path: path to load a Pandas DataFrame.
    :return: a preprocessed Pandas DataFrame
    """

    df = pd.read_csv(path)

    df = df.loc[(df['metype'] <= 5), ['fromrun', 'fromlumi', 'hname', 'histo',
                                      'entries', 'Xmax', 'Xmin', 'Xbins']]

    # eval/change the string and cut the first and last value which correspond to Underflow and Overflow
    df['histo'] = df['histo'].apply(literal_eval).apply(lambda x: x[1:-1])

    # assigning row indices w.r.t the two columns "fromrun" and "fromlumi"
    df.set_index(['fromrun', 'fromlumi'], inplace=True, drop=False)
    df.sort_index(axis=0, inplace=True)
    return df


def check_ls(json_data, run, ls):

    """
    Determines whether the LS is a good LS (1) or a bad LS 0
    :param run: int, indicating number of run
    :param ls: int, lumisection number
    :return: int, a binary-valued label of the lumisection
    """
    is_ok = 0

    if str(run) in json_data.keys():
        for i in json_data[str(run)]:
            if i[0] <= ls <= i[1]:
                is_ok = 1
                return is_ok
    return is_ok


def convert_df_to_np_array_per_a_histogram(df, name_of_histo, selected_run):

    """
    loads a preprocessed DataFrame and returns:
      1) a numpy array of corresponding histograms which is determined by the argument "name_of_histo";

    :param df: A Pandas DataFrame, A preprocessed Pandas DataFrame
    :param name_of_histo: string, name of histogram which one decides to use for further analysis
    :param selected_run: list, list of selected run(s) to create a numpy array
    :return: Numpy array, a numpy array of corresponding histograms for a selected run(s)
    """

    # load the golden json file
    global histos
    global labels
    json_data = {}

    with open('/home/fratnikov/cms/ml4dc/ML_2020/Scripts2020/GoldenJSON17.json') as json_file:
        json_data = json.load(json_file)

    for run in df['fromrun'].unique():

        if run in selected_run:
            print("run:", run)
            df_of_selected_run = df.loc[(df['fromrun'] == run) & (df['hname'] == name_of_histo)]
            for i, row in df_of_selected_run.iterrows():
                ls = row['fromlumi']
                a_histo = row['histo']
                label = check_ls(json_data=json_data, run=run, ls=ls)
                if label == 1:
                    a_histo = a_histo.split("[")[1].split("]")[0]
                    a_histo = a_histo.split(", ")
                    a_histo = [int(i) for i in a_histo]
                    dim = len(a_histo)
                    break

            print("feature's len:", dim)

            histos = np.array([]).reshape(dim, 0)
            labels = np.array([]).reshape(1, 0)

            for i, row in df_of_selected_run.iterrows():
                ls = row['fromlumi']
                a_histo = row['histo']  # later I may need to modify this so that it is applicable for list of list
                label = check_ls(json_data=json_data, run=run, ls=ls)
                a_histo = a_histo.split("[")[1].split("]")[0]
                a_histo = a_histo.split(", ")
                a_histo = [int(i) for i in a_histo]
                histos = np.c_[histos, a_histo]
                labels = np.c_[labels, label]
                # histos.append(a_histo)

            histos = histos.T
            labels = labels.T
            print("histos.shape:", histos.shape, labels.shape)

    return histos, labels


def train_val_test_splitter(Xg, Xb, n_repeats=5, settings=[(0.98, 0.02)], ):

    DATA = {}

    for setting in settings:

        DATA[setting] = {}

        for repeat in range(n_repeats):
            DATA[setting][repeat] = {}

            Lg = np.repeat(int(1), Xg.shape[0])  # Labels Good

            Xg_train, Xg_test, Lg_train, Lg_test = train_test_split(Xg, Lg,
                                                                    test_size=setting[-1],
                                                                    shuffle=True)

            Xg_test, Xg_val, Lg_test, Lg_val = train_test_split(Xg_test, Lg_test,
                                                                test_size=0.5,
                                                                shuffle=True)

            Lb = np.repeat(int(0), Xb.shape[0])  # Labels Bad

            Xb_train, Xb_test, Lb_train, Lb_test = train_test_split(Xb, Lb,
                                                                    test_size=setting[-1],
                                                                    shuffle=True)

            Xb_test, Xb_val, Lb_test, Lb_val = train_test_split(Xb_test, Lb_test,
                                                                test_size=0.5,
                                                                shuffle=True)

            X_train = np.concatenate((Xg_train, Xb_train), axis=0)
            X_val = np.concatenate((Xg_val, Xb_val), axis=0)
            X_test = np.concatenate((Xg_test, Xb_test), axis=0)

            L_train = np.concatenate((Lg_train, Lb_train), axis=0)
            L_val = np.concatenate((Lg_val, Lb_val), axis=0)
            L_test = np.concatenate((Lg_test, Lb_test), axis=0)

            print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

            DATA[setting][repeat]['X_tr'] = X_train
            DATA[setting][repeat]['X_vl'] = X_val
            DATA[setting][repeat]['X_ts'] = X_test
            DATA[setting][repeat]['y_tr'] = L_train
            DATA[setting][repeat]['y_vl'] = L_val
            DATA[setting][repeat]['y_ts'] = L_test

    return DATA

    # with open (os.path.join('SANC_computation', name+features_type+str(size)+'.pickle'), 'wb') as fp:
    #     pickle.dump(data, fp)


if __name__ == '__main__':

    path_to_store = '/home/sshalileh/ml4dc/matrices'

    parser = argparse.ArgumentParser()

    parser.add_argument('--Selected_run', type=str, default='297050', help='Enter run number')
    parser.add_argument('--Name_of_histo', type=str, default='goodvtxNbr', help='Name of the histogram '
                                                                                'for creating the Numpy array')

    args = parser.parse_args()

    selected_run, name_of_histo = parse_argument(args=args)

    """ 
    # Loading all the stored DataFrames
    all_paths = glob.glob('/home/fratnikov/cms/ml4dc/ML_2020/UL2017_Data/*_1D_*/*.csv')

    print("len of loaded files:", len(all_paths))

    df_list = []
    for path in all_paths:
        df_list.append(load_process_a_df(path=path))

    # Concatenating all 1D processed DataFrames into one DataFrame
    df = pd.concat(df_list, axis=0, ignore_index=True)
    """
    # once I run the chunk of code above and because it very time consuming I decided to save the df ...
    df = pd.read_csv("/home/sshalileh/ml4dc/matrices/all_runs_df.csv")

    print(df.head())

    for histogram_name in name_of_histo:
        print("histogram:", histogram_name)

        numpy_array, labels = convert_df_to_np_array_per_a_histogram(df=df, name_of_histo=histogram_name,
                                                                     selected_run=selected_run)

        np.savetxt(os.path.join(path_to_store,
                                str(selected_run[0]) + "-" + histogram_name + '.npy'), numpy_array)

        np.savetxt(os.path.join(path_to_store,
                                str(selected_run[0]) + "-" + histogram_name + '-labels.npy'), labels)

        Xg = numpy_array[np.where(labels == 1), :]
        Xb = numpy_array[np.where(labels == 0), :]
        Xg = Xg[0]
        Xb = Xb[0]

        print("before split:", Xg.shape, Xb.shape)
        print(" ")

        data = train_val_test_splitter(Xg=Xg, Xb=Xb, settings=[(0.98, 0.02), ])

        with open(os.path.join(path_to_store,
                               str(selected_run[0]) + "-" + histogram_name + '.pickle'), 'wb') as fp:
            pickle.dump(data, fp)




