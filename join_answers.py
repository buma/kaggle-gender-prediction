# -*- coding: utf-8 -*-
"""It joins training file with answers"""
from data_io import DataIO
import pandas as pd
from os.path import join as path_join

dio = DataIO("Settings_submission.json")

training_data = pd.read_csv(
    path_join(dio.data_dir, "data", "raw", "train.csv")
)

data_path = dio.train_file.split("/")[:-1]

filename = path_join(dio.data_dir, "data", "raw", "train_answers.csv")

training_answers = pd.read_csv(filename)

train_with_answers = pd.merge(training_data, training_answers, on='writer')

#train_with_answers.to_csv(
    #path_join(dio.data_dir, "data", "processed", "train_w_answers.csv")
#)

store = pd.HDFStore(
    path_join(dio.data_dir, "data", "processed", "train_w_answers.h5")
)

store["train_w_answers"] = train_with_answers

store.close()
