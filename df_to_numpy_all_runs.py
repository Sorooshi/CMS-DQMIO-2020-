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
    name_of_histo = args.Name_of_histo
    name_of_histo = name_of_histo.split(", ")
    return name_of_histo


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


def convert_df_to_np_array_per_a_histogram(df, name_of_histo,):

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
        break

    print("feature's len:", dim)

    histos = np.array([]).reshape(dim, 0)
    labels = np.array([])
    indices = []

    for run in df['fromrun'].unique():
        print("run:", run)
        df_of_selected_run = df.loc[(df['fromrun'] == run) & (df['hname'] == name_of_histo)]
        for i, row in df_of_selected_run.iterrows():
            ls = row['fromlumi']
            a_histo = row['histo']  # later I may need to modify this so that it is applicable for list of list
            label = check_ls(json_data=json_data, run=run, ls=ls)
            a_histo = a_histo.split("[")[1].split("]")[0]
            a_histo = a_histo.split(", ")
            a_histo = [int(i) for i in a_histo]
            indices.append((run, ls))
            histos = np.c_[histos, a_histo]
            labels = np.r_[labels, label]

    histos = histos.T
    labels = labels.T
    print("histos.shape:", histos.shape, labels.shape)

    return histos, labels, indices


def train_val_test_splitter(Xg, Xb, Ig, Ib, n_repeats=2, settings=[(0.98, 0.02)], ):

    DATA = {}

    for setting in settings:

        DATA[setting] = {}

        for repeat in range(n_repeats):
            DATA[setting][repeat] = {}

            Lg = np.repeat(int(1), Xg.shape[0])  # Labels Good

            pos_test_val_indices = np.random.choice(np.arange(0, Xg.shape[0]),
                                                    size=int(Xg.shape[0]*setting[-1]/100),
                                                    replace=False)

            pos_train_indices = set(np.arange(0, Xg.shape[0])).difference(set(pos_test_val_indices))

            pos_test_indices = np.random.choice(pos_test_val_indices,
                                                size=int(len(pos_test_val_indices)/2),
                                                replace=True
                                                )

            pos_val_indices = set(pos_test_val_indices).difference(set(pos_test_indices))

            Xg_train = Xg[pos_train_indices, :]
            Lg_train = Lg[pos_train_indices]
            Ig_train = Ig[pos_train_indices]

            Xg_test = Xg[pos_test_indices, :]
            Lg_test = Lg[pos_test_indices]
            Ig_test = Ig[pos_train_indices]

            Xg_val = Xg[pos_val_indices, :]
            Lg_val = Lg[pos_val_indices]
            Ig_val = Ig[pos_train_indices]

            Lb = np.repeat(int(0), Xb.shape[0])  # Labels Bad
            neg_test_val_indices = np.random.choice(np.arange(0, Xb.shape[0]),
                                                    size=int(Xb.shape[0] * setting[-1] / 100),
                                                    replace=False)

            neg_train_indices = set(np.arange(0, Xb.shape[0])).difference(set(neg_test_val_indices))

            neg_test_indices = np.random.choice(neg_test_val_indices,
                                                size=int(len(neg_test_val_indices)/2),
                                                replace=False
                                                )

            neg_val_indices = set(neg_test_val_indices).difference(set(neg_test_indices))

            Xb_train = Xb[neg_train_indices, :]
            Lb_train = Lb[neg_train_indices]
            Ib_train = Ib[neg_train_indices]

            Xb_test = Xb[neg_test_indices, :]
            Lb_test = Lb[neg_test_indices]
            Ib_test = Ib[neg_train_indices]

            Xb_val = Xb[neg_val_indices, :]
            Lb_val = Lb[neg_val_indices]
            Ib_val = Ib[neg_train_indices]

            X_train = np.concatenate((Xg_train, Xb_train), axis=0)
            X_val = np.concatenate((Xg_val, Xb_val), axis=0)
            X_test = np.concatenate((Xg_test, Xb_test), axis=0)

            L_train = np.concatenate((Lg_train, Lb_train), axis=0)
            L_val = np.concatenate((Lg_val, Lb_val), axis=0)
            L_test = np.concatenate((Lg_test, Lb_test), axis=0)

            I_train = np.concatenate((Ig_train, Ib_train), axis=0)
            I_val = np.concatenate((Ig_val, Ib_val), axis=0)
            I_test = np.concatenate((Ig_test, Ib_test), axis=0)

            print("X_train:", X_train.shape, "X_val:", X_val.shape, "X_test:", X_test.shape)

            DATA[setting][repeat]['X_tr'] = X_train
            DATA[setting][repeat]['X_vl'] = X_val
            DATA[setting][repeat]['X_ts'] = X_test
            DATA[setting][repeat]['y_tr'] = L_train
            DATA[setting][repeat]['y_vl'] = L_val
            DATA[setting][repeat]['y_ts'] = L_test

            DATA[setting][repeat]['I_tr'] = I_train
            DATA[setting][repeat]['I_vl'] = I_val
            DATA[setting][repeat]['I_ts'] = I_test

    return DATA


if __name__ == '__main__':

    path_to_store = '/home/sshalileh/ml4dc/matrices'

    parser = argparse.ArgumentParser()

    parser.add_argument('--Selected_run', type=str, default='297050', help='Enter run number')
    parser.add_argument('--Name_of_histo', type=str, default='goodvtxNbr', help='Name of the histogram '
                                                                                'for creating the Numpy array')

    args = parser.parse_args()

    # name_of_histo = parse_argument(args=args)

    # Loading all the stored DataFrames
    all_paths = glob.glob('/home/fratnikov/cms/ml4dc/ML_2020/UL2017_Data/*_1D_*/*.csv')

    print("len of loaded files:", len(all_paths))

    # df_list = []
    # for path in all_paths:
    #     df_list.append(load_process_a_df(path=path))
    #
    # # Concatenating all 1D processed DataFrames into one DataFrame
    # df = pd.concat(df_list, axis=0, ignore_index=True)
    # df.to_csv("/home/sshalileh/ml4dc/matrices/all_runs_df.csv")

    # # once I run the chunk of code above and because it very time consuming I decided to save the df ...
    df = pd.read_csv("/home/sshalileh/ml4dc/matrices/all_runs_df.csv")

    print(df.head())

    name_of_histo = df['hname'].unique()

    for histogram_name in name_of_histo:

        print("histogram:", histogram_name)

        numpy_array, labels, indices = convert_df_to_np_array_per_a_histogram(df=df, name_of_histo=histogram_name,)

        np.savetxt(os.path.join(path_to_store, histogram_name + '.npy'), numpy_array)

        np.savetxt(os.path.join(path_to_store, histogram_name + '-labels.npy'), labels)

        # Xg = numpy_array[np.where(labels == 1), :]
        # Xb = numpy_array[np.where(labels == 0), :]
        # Ig = indices[np.where(labels == 1)]
        # Ib = indices[np.where(labels == 0)]
        # Xg = Xg[0]
        # Xb = Xb[0]
        # Ig = Ig[0]
        # Ib = Ib[0]
        #
        # print("before split:", Xg.shape, Xb.shape)
        # print(" ")
        #
        # data = train_val_test_splitter(Xg=Xg, Xb=Xb, Ig=Ig, Ib=Ib, settings=[(0.98, 0.02), ])
        #
        # with open(os.path.join(path_to_store, histogram_name + '.pickle'), 'wb') as fp:
        #     pickle.dump(data, fp)




