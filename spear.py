from data_io import DataIO
import pandas as pd
import ramp as rp
import joblib
from os.path import join as path_join
import numpy as np
from itertools import islice
import sklearn


dio = DataIO("Settings_submission.json")

store = pd.HDFStore(dio.train_file, "r")

submit = False

if submit:
    train_id = "train_w_answers"
    test_id = "test"
    store_path = dio.cache_dir + "_submit"
else:
    train_id = "train_train"
    test_id = "train_test"
    store_path = dio.cache_dir + "1"


training_data = store[train_id]
writer = training_data.writer

context = rp.DataContext(
    store=store_path,
    data=training_data
)


configs = joblib.load(path_join(dio.cache_dir, "all_scores_all_vse"))
# config_id = 84
config_id = 30
config_log = configs[config_id][0]

#print str(config_log)

my_cv = list(islice(sklearn.cross_validation.LeavePLabelOut(writer, p=75), 10))  # 10 splits


def main(job_id, params):

    print "Job id:", str(job_id)
    n_trees = params["ntrees"][0]
    max_features = params["max_f"][0]
    criterion = params["criterion"][0]
    config_log.model.estimator.set_params(n_estimators=n_trees)
    config_log.model.estimator.set_params(max_features=max_features)
    config_log.model.estimator.set_params(criterion=criterion)

    #print config_log
    print params
    scores = rp.models.cv(config_log, context, folds=my_cv, repeat=2,
                print_results=True)
    return np.array(scores['logloss']).mean()
