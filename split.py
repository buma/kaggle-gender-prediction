'''
split a file into train test and validation set on writer
Usage: split.py'
'''

import random
from data_io import DataIO
import os
import pandas as pd
from os.path import join as path_join

dio = DataIO("Settings_submission.json")
dio1 = DataIO("Settings.json")

#input_file = dio.train_file
#output_file1 = dio1.train_file
#output_file2 = dio1.valid_file

#print "Input: %s " % input_file
#print "Train file: %s " % output_file1
#print "Validation file: %s " % output_file2

store = pd.HDFStore(
    path_join(dio.data_dir, "data", "processed", "train_w_answers.h5")
)

train = store["train_w_answers"]
print train
random.seed(42)
all_writers = train.writer.unique()
print "all writers:", len(all_writers)
random.shuffle(all_writers)
#first 80%
first = int(round(len(all_writers) * 0.8))
train_writers = all_writers[0:first]
valid_writers = all_writers[first:]

print "Train subjects:", len(train_writers)
print "Valid subjects:", len(valid_writers)

run = raw_input("OK (Y/N)?")
print run
if run != "Y":
    os.exit()

del store["train_train"]

del store["train_test"]

train_train = train[train['writer'].isin(train_writers)].copy()
train_train.index = range(len(train_train.index))
store["train_train"] = train_train

train_test = train[train['writer'].isin(valid_writers)].copy()
train_test.index = range(len(train_test.index))
store["train_test"] = train_test

print store

store.close()
