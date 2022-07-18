"""Main script to start pipeline is divided into three components: data preprocessing, model training, and model testing"""
import yaml
import os
import numpy as np
import pandas as pd
import shutil
import pickle
from datetime import datetime
from shutil import copy
import torch
from sklearn.model_selection import train_test_split

from process_bag import reorder_bag, extract_bag_to_csv
from preprocess import DataClass
from utils import  state_variable_plots, state_der_plots


def preprocess_data(args):
    # assumes rosbag data has already been recorded
    # e.g. rosbag record /chassisState /ground_truth/state_transformed /ground_truth/state /ground_truth/state_raw /clock /tf /imu/imu /wheelSpeeds /joy --duration=60
    print("Preprocessing data...")

    # make dir to store preprocessed data, plots, and rosbag files
    data_dir = 'data'
    plots_dir = 'preprocess_plots'
    rosbag_dir = 'rosbag_files'
    dirs = [data_dir, plots_dir, rosbag_dir]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)

    # reorder bag file based on header timestamps
    reorder_bag(args["rosbag_filepath"])

    # specify topics to extract from bag
    topics = [topic['name'] for topic in args['topics']]

    # extract specified topics from rosbag
    topic_file_paths = extract_bag_to_csv(args["rosbag_filepath"], topics=topics, folder=rosbag_dir)

    # list to keep track of all preprocessed data dfs
    data_dfs = []

    # variable for the final time step
    end_point = None
    # variable to keep track of resulting sampling rate
    sample_rate = None

    for topic_args in args["topics"]:
        print("\nPreprocessing topic %s..." % topic_args["name"])
        # init DataClass object
        data_obj = DataClass(topic_name=topic_args["name"], column_mapper=topic_args["col_mapper"], plot_folder=plots_dir,
                             df_file_path=topic_file_paths[topic_args['name']], make_plots=args['make_preprocessing_plots'])

        # prep data: load csv as df and rename columns
        print("Loading csv file and renaming columns..")
        data_obj.prep_data()

        # check if need to trim sequence to a specified time in seconds
        if args['total_data']:
            print("Trimming data to %.0f seconds..." % args['total_data'])
            data_obj.trim_sequence(args['total_data'] + round(data_obj.df.head(1)["time"].values[0]))

        # check if need to convert quaternion to euler angles (roll, pitch, yaw)
        if 'quaternion_to_euler' in topic_args:
            print("Converting quaternion to euler angle...")
            x, y, z, w = topic_args['quaternion_to_euler'].values()
            data_obj.df = DataClass.convert_quaternion_to_euler(data_obj.df, x, y, z, w)

        # check if need to compute derivatives from data
        if 'compute_derivatives' in topic_args:
            print("Computing derivative data...")
            der = topic_args['compute_derivatives']
            data_obj.get_data_derivative(cols=der['cols'], degree=der['degree'])

        # init endpoint
        if end_point is None:
            # resample_data assumes time data starts at 0 so need to shift the sequence by setting end_point = max_time - min_time
            end_point = int(round(data_obj.df.tail(1)["time"].values[0]) - round(data_obj.df.head(1)["time"].values[0]))

        # check if need to resample data
        resample = topic_args['resample']
        if resample['cols']:
            print("Resampling data...")
            # if up/down sampling factor not specified resample data to match sampling rate of other data
            if not resample['upsampling_factor']:
                up = sample_rate
                down = len(data_obj.df)
            else:
                up = resample['upsampling_factor']
                down = resample['downsampling_factor']
            data_obj.resample_data(end_point, up, down, resample['cols'])
            sample_rate = len(data_obj.df)

        # check if need to truncate columns to min and max
        if 'trunc' in topic_args:
            print("Truncating data...")
            trunc = topic_args['trunc']
            data_obj.trunc(trunc['cols'], maximum=trunc['max'], minimum=trunc['min'])

        # save state data to disk
        print("Saving to disk...")
        data_obj.df.to_csv(os.path.join(data_dir, topic_args['filename']), index=False)

        data_dfs.append(data_obj.df)

    # merge control and state data
    final = pd.concat(data_dfs, axis=1)

    # generate state vs. time and trajectory plot for preprocessed data
    state_variable_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                         cols_to_include=np.concatenate((args['state_cols'], args['ctrl_cols'])))

    # generate state der vs. time plots
    state_der_plots(df1=final, df1_label="preprocessed ground truth", dir_path="preprocess_plots/",
                    cols_to_include=np.concatenate((args['label_cols'], args['ctrl_cols'])))

    # check if standardize data option is set to true
    if args["standardize_data"]:
        print("\nStandardizing data...")
        # standardize features and labels
        final, scaler_list = DataClass.standardize_data(final, plots_dir, args["feature_cols"], args["label_cols"])

        feature_scaler, label_scaler = scaler_list
        # add to args dict scaler objects
        args["feature_scaler"] = feature_scaler
        args["label_scaler"] = label_scaler

        # save scaler objects to disk as pickles
        pickle.dump(feature_scaler, open(data_dir + "/feature_scaler.pkl", "wb"))
        pickle.dump(label_scaler, open(data_dir + "/label_scaler.pkl", "wb"))

    # save to disk
    final.to_csv(os.path.join(data_dir, args["final_file_name"]), index=False)

    # move files to a common folder
    folder = "pipeline_files/" + args["run_name"]
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        # prompt user if directory already exists
        answer = None
        while not(answer == "y" or answer == "n"):
            answer = raw_input("Replace already existing directory %s? (y/n): " % folder).lower().strip()
            print("")
        if answer == "y":
            shutil.rmtree(folder)
            os.makedirs(folder)
        else:
            print("Keeping old directory and leaving preprocessing files in working directory %s..." % os.getcwd())
            exit(0)

    # move files
    for d in dirs:
        shutil.move(d, folder)

    print("Done preprocessing data")


def main():
    # load config file into args
    config = "./config.yml"
    with open(config, "r") as yaml_file:
        args = yaml.load(yaml_file, Loader=yaml.FullLoader)
    args['config_file'] = config

    options = ["preprocess_data", "train_model", "test_model"]
    if not any([args[i] for i in options if i in args.keys()]):
        print("No option has been selected!")
        print("One of %s needs to be set to True in config.yml" % str(options))
        exit(1)

    if args["preprocess_data"]:
        preprocess_data(args)


if __name__ == '__main__':
    main()