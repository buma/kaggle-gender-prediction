"""Gets feature name from prefixes from:https://www.kaggle.com/c/icdar2013-gender-prediction-from-handwriting/forums/t/3982/looking-for-a-feature"""
template = "{0} = get_columns_with_prefix('{1}')"
f = open("prefixs")

features = []
for line in f.read().splitlines():
    feature_name, prefix = line.split(":")
    print template.format(feature_name, prefix)
    features.append(prefix)

print features
