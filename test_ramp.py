from data_io import DataIO
import pandas as pd
import ramp as rp
from ramp.estimators.sk import BinaryProbabilities
import sklearn
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import joblib
from os.path import join as path_join
import numpy as np
from itertools import islice


dio = DataIO("Settings_submission.json")

store = pd.HDFStore(dio.train_file, "r")

print store

training_data = store["train_train"]
writer = training_data.writer
#removes constant columns
non_constant_cols = (training_data.std() > 1e-9)
all_columns = non_constant_cols.index[non_constant_cols.values]
#all_columns = training_data.columns

learn = True




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

#sums how many columns are chosen with these features
print map(lambda i: len(eval('f%d' % i)), range(1, 27))
print sum(map(lambda i: len(eval('f%d' % i)), range(1, 27)))

#prints unused columns
#features = ['DirectionPerpendicular5Hist10', 'CurvatureAli5Hist100', 'tortuosityDirectionHist10', 'chaincodeHist_8', 'chaincode8order2_64', 'chaincode8order3_512', 'chaincode8order4_4096', 'directions_hist1_4', 'directions_hist2_8', 'directions_hist3_12', 'directions_hist4_16', 'directions_hist5_20', 'directions_hist6_24', 'directions_hist7_28', 'directions_hist8_32', 'directions_hist9_36', 'directions_hist10_40', 'directions_hist1a2_12', 'directions_hist1a2a3_24', 'directions_hist1a2a3a4_40', 'directions_hist1a2a3a4a5_60', 'directions_hist1a2a3a4a5a6_84', 'directions_hist1a2a3a4a5a6a7_112', 'directions_hist1a2a3a4a5a6a7a8_144', 'directions_hist1a2a3a4a5a6a7a8a9_180', 'directions_hist1a2a3a4a5a6a7a8a9a10_220']
#for column in all_columns:
    #nex = False
    #for feature in features:
        #if column.startswith(feature):
            #nex = True
            #break
    #if nex:
        #continue
    #print column

#a=5/0

print training_data

context = rp.DataContext(
    store=dio.cache_dir,
    data=training_data
)

base_config = rp.Configuration(
    target=rp.AsFactor('male'),
    metrics=[rp.metrics.LogLoss()],
)

#Writer language factor, same text factor
base_features = [
    rp.Feature("writer"),
    rp.AsFactor("language"),
    rp.AsFactor("same_text")
    #rp.FillMissing(f, 0) for f in training_data.columns[3:-1]
]

f6_features = [rp.FillMissing(f, 0) for f in f6]
f7_features = [rp.FillMissing(f, 0) for f in f7]
f26_features = [rp.FillMissing(f, 0) for f in f26]
all_f = [rp.FillMissing(f, 0) for f in all_columns[3:-1]]

f6_f7 = list(f6_features)
f6_features.extend(f7_features)
f6_f7_f26 = list(f6_f7)
f6_f7_f26.extend(f26_features)

factory = rp.ConfigFactory(
    base_config,
    features=[
        #('BASE: writer, F_language, F_same text', base_features),
        #('BASE + f6', f6_features),
        #('BASE + f7', f7_features),
        #('BASE + f26', f26_features),
        ('BASE + f6 + f7', f6_f7),
        ('BASE + f6 + f7 + f26', f6_f7_f26),
        ('BASE + all not only f', all_f),
        #('BASE + f6 norm', [rp.Normalize(f) for f in f6_features]),
        #('BASE + f7 norm', [rp.Normalize(f) for f in f7_features]),
        #('BASE + f26 norm', [rp.Normalize(f) for f in f26_features]),
        ('BASE + f6 + f7 norm', [rp.Normalize(f) for f in f6_f7]),
        ('BASE + f6 + f7 + f26 norm', [rp.Normalize(f) for f in f6_f7_f26]),
        ('BASE + all not only f norm', [rp.Normalize(f) for f in all_f]),

        #('all except subject top 100 with RF', [rp.trained.FeatureSelector(
            #base_features,
         ##use random forest to trim features
            #rp.selectors.RandomForestSelector(classifier=True),
            #target=rp.AsFactor('activity'),  # target to use
            #n_keep=100,
         #)]),
        #('all except subject normalized top 100 with RF', [rp.trained.FeatureSelector(
            #normalized_base_features,
##use random forest to trim features
            #rp.selectors.RandomForestSelector(classifier=True),
            #target=rp.AsFactor('activity'),  # target to use
            #n_keep=100,
        #)]),
    ],
    model=[
        BinaryProbabilities(
            sklearn.linear_model.LogisticRegression(random_state=42)),
        BinaryProbabilities(
            sklearn.linear_model.LogisticRegression(random_state=42, penalty='l1')),
        BinaryProbabilities(
            sklearn.linear_model.SGDClassifier(random_state=42, loss='log')),
        BinaryProbabilities(
            sklearn.linear_model.SGDClassifier(random_state=42, loss='modified_huber')),
        BinaryProbabilities(
            sklearn.ensemble.RandomForestClassifier(random_state=42, n_jobs=4, n_estimators=20)),
        BinaryProbabilities(
            sklearn.ensemble.ExtraTreesClassifier(random_state=42, n_jobs=4, n_estimators=20)),
        BinaryProbabilities(
            sklearn.ensemble.AdaBoostClassifier()),
        BinaryProbabilities(
            sklearn.ensemble.AdaBoostClassifier(n_estimators=100)),
    ]
)

def predict(config, context):
    test = store["train_test"]
    ctx = context.copy()
    train_idx = ctx.data.index
    ctx.data = ctx.data.append(test, ignore_index=True)
    print ctx.data
    ctx.train_index = train_idx
    max_idx = max(train_idx)
    test_idx = range(max_idx + 1, max(ctx.data.index)+1)
    preds, predict_x, predict_y = rp.models.predict(
            config,
            ctx,
            test_idx
            )
    actuals = predict_y.reindex(test_idx)
    scores = []
    print actuals[:10]
    print preds[:10]
    score = metric.score(actuals, preds)
    print "%0.4f" % score

    return preds, train.writer

if learn:
#my_cv = sklearn.cross_validation.LeaveOneLabelOut(writer)
    #my_cv = list(islice(sklearn.cross_validation.LeavePLabelOut(writer, p=45), 5))  # 5 splits
    my_cv = list(islice(sklearn.cross_validation.LeavePLabelOut(writer, p=75), 3))  # 3 splits

    all_scores = []
#myStore = rp.store.PickleStore(path=dio.cache_dir)
    for config in factory:
        #print
        #print "S", str(config)
        #ctx = context.copy()
        #for train, test in my_cv:
            #ctx.train_index = train
            #key = rp.models.get_key(config, ctx)
            #safe_name = myStore.safe_name(key)
            #print "\tKE:", ctx.create_key()
            #print key
            #print "\tSN:",safe_name
            #ven = myStore.load(key)
            #print "\t",str(ven)
        scores = rp.models.cv(config, context, folds=my_cv, repeat=2,
                    print_results=True)
        all_scores.append((config, scores))
        joblib.dump(all_scores, path_join(dio.cache_dir, "all_scores_all"))
        #break

    joblib.dump(all_scores, path_join(dio.cache_dir, "all_scores_all"))
else:
    configs = joblib.load(path_join(dio.cache_dir, "all_scores_all_vse"))
    #config_id = 84
    config_id = 90
    config_log = configs[config_id][0]

    print str(config_log)

    preds, writers = predict(config_log, context)

    print preds



##map(lambda conf, scores: print "\n", str(conf); rp.models.print_scores(scores), all_scores)
#joblib.dump(all_scores, path_join(dio.cache_dir, "all_scores_RF1"))

#print "scores"
#for config, scores in all_scores:
    #print "\n", config
    #rp.models.print_scores(scores)
