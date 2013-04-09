from data_io import DataIO
import joblib
from os.path import join as path_join
import numpy as np
import ramp as rp
from itertools import islice

dio = DataIO("Settings_submission.json")

all_scores_old = joblib.load(path_join(dio.cache_dir, "all_scores_all3"))
#all_scores_old1 = joblib.load(path_join(dio.cache_dir, "all_scores_all1"))
#all_scores_old2 = joblib.load(path_join(dio.cache_dir, "all_scores_all2"))

#all_scores_old.extend(all_scores_old1)
#all_scores_old.extend(all_scores_old2)

joblib.dump(all_scores_old, path_join(dio.cache_dir, "all_scores_all_vse"))

all_scores = []
for i, (config, scores) in enumerate(all_scores_old):
    all_scores.append((config, scores, i))



def sort_mean(element):
    return np.array(element[1]['logloss']).mean()


def sort_std(element):
    return np.array(element[1]['logloss']).std()


def sort_min(element):
    return min(element[1]['logloss'])

mean_scores = sorted(all_scores, key=sort_mean, reverse=False)
std_scores = sorted(all_scores, key=sort_std, reverse=False)
min_scores = sorted(all_scores, key=sort_min, reverse=False)

n = 10

print "Best %d sorted by mean increasing" % n
for config, scores, idx in islice(mean_scores, n):
    print idx, "\n", config
    print scores
    rp.models.print_scores(scores)
n = 5
print "Best %d sorted by std increasing" % n
for config, scores, idx in islice(std_scores, n):
    print idx, "\n", config
    rp.models.print_scores(scores)

print "Best %d sorted by min decreasing" % n
for config, scores, idx in islice(min_scores, n):
    print idx, "\n", config
    rp.models.print_scores(scores)
