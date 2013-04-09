from data_io import DataIO
import pandas as pd
import ramp as rp
import joblib
from os.path import join as path_join
import numpy as np


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
# removes constant columns
non_constant_cols = (training_data.std() > 1e-9)
all_columns = non_constant_cols.index[non_constant_cols.values]
# all_columns = training_data.columns


def get_columns_with_prefix(prefix):
    """Returns only column names that starts with prefix"""
    return filter(lambda column: column.startswith(prefix), all_columns)

f1 = get_columns_with_prefix('DirectionPerpendicular5Hist10')
f2 = get_columns_with_prefix('CurvatureAli5Hist100')
f3 = get_columns_with_prefix('tortuosityDirectionHist10')
f4 = get_columns_with_prefix('chaincodeHist_8')
f5 = get_columns_with_prefix('chaincode8order2_64')
f6 = get_columns_with_prefix('chaincode8order3_512')
f7 = get_columns_with_prefix('chaincode8order4_4096')
f8 = get_columns_with_prefix('directions_hist1_4')
f9 = get_columns_with_prefix('directions_hist2_8')
f10 = get_columns_with_prefix('directions_hist3_12')
f11 = get_columns_with_prefix('directions_hist4_16')
f12 = get_columns_with_prefix('directions_hist5_20')
f13 = get_columns_with_prefix('directions_hist6_24')
f14 = get_columns_with_prefix('directions_hist7_28')
f15 = get_columns_with_prefix('directions_hist8_32')
f16 = get_columns_with_prefix('directions_hist9_36')
f17 = get_columns_with_prefix('directions_hist10_40')
f18 = get_columns_with_prefix('directions_hist1a2_12')
f19 = get_columns_with_prefix('directions_hist1a2a3_24')
f20 = get_columns_with_prefix('directions_hist1a2a3a4_40')
f21 = get_columns_with_prefix('directions_hist1a2a3a4a5_60')
f22 = get_columns_with_prefix('directions_hist1a2a3a4a5a6_84')
f23 = get_columns_with_prefix('directions_hist1a2a3a4a5a6a7_112')
f24 = get_columns_with_prefix('directions_hist1a2a3a4a5a6a7a8_144')
f25 = get_columns_with_prefix('directions_hist1a2a3a4a5a6a7a8a9_180')
f26 = get_columns_with_prefix('directions_hist1a2a3a4a5a6a7a8a9a10_220')

# sums how many columns are chosen with these features
print map(lambda i: len(eval('f%d' % i)), range(1, 27))
print sum(map(lambda i: len(eval('f%d' % i)), range(1, 27)))

context = rp.DataContext(
    store=store_path,
    data=training_data
)

base_config = rp.Configuration(
    target=rp.AsFactor('male'),
    metrics=[rp.metrics.LogLoss()],
)

# Writer language factor, same text factor
base_features = [
    rp.Feature("writer"),
    rp.AsFactor("language"),
    rp.AsFactor("same_text")
    # rp.FillMissing(f, 0) for f in training_data.columns[3:-1]
]

f6_features = [rp.FillMissing(f, 0) for f in f6]
f7_features = [rp.FillMissing(f, 0) for f in f7]
f26_features = [rp.FillMissing(f, 0) for f in f26]
all_f = [rp.FillMissing(f, 0) for f in all_columns[3:-1]]

f6_f7 = list(f6_features)
f6_features.extend(f7_features)
f6_f7_f26 = list(f6_f7)
f6_f7_f26.extend(f26_features)


def predict(config, context):
    test = store[test_id]
    ctx = context.copy()
    train_idx = ctx.data.index
    ctx.data = ctx.data.append(test, ignore_index=True)
    print ctx.data
    ctx.train_index = train_idx
    max_idx = max(train_idx)
    test_idx = range(max_idx + 1, max(ctx.data.index) + 1)
    preds, predict_x, predict_y = rp.models.predict(
        config,
        ctx,
        test_idx
    )
    if not submit:
        actuals = predict_y.reindex(test_idx)
        score = config.metrics[0].score(actuals, preds)
        print "Score: %0.4f" % score
        print actuals[:10]
    print preds[:10]

    return preds, test.writer

configs = joblib.load(path_join(dio.cache_dir, "all_scores_all_vse"))
# config_id = 84
config_id = 8
config_log = configs[config_id][0]

print str(config_log)

preds, writers = predict(config_log, context)

if submit:
    filename = path_join(dio.submission_path, "SGD_modified_huber_all.csv")

    df = pd.DataFrame({
        'writer': writers.values,
        'male': preds.values
        })

    grouped = df.groupby('writer')
    avg = grouped.aggregate(np.mean)

    print avg

    avg.to_csv(filename, header=False)

