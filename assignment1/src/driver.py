from utils import *

# Load dataframes into metadata map
for key in metadata:
    print(f'key: {key}')
    
    metadata[key]["dataframe"] = loadData(metadata[key])
    metadata[key]["dataframe"] = handleMissingValues(metadata[key])
    metadata[key]["dataframe"] = encodeOrdinalFeatures(metadata[key])
    metadata[key]["dataframe"] = encodeNominalFeatures(metadata[key])
    metadata[key]["dataframe"] = discretization(metadata[key], False)
    # metadata[key]["dataframe"].to_csv('test.csv', sep="\t", index=False)
    train, test = randomPartition(metadata[key])

    train, test = zScore(train, test, metadata[key])

    # metadata[key]["folds"] = partition(metadata[key])
    print(f'training set length: {len(train)}')
    # print(f'folds: {stratify_k_fold(train, metadata[key]["classColumn"], 2)}')
    # print(f'prediction: {}')
    if key == 'forest-fires':
        null_model_predictor_regression(train, metadata[key]["regressionColumn"]).to_csv('regression.csv', sep=",", index=False)
    else:
        null_model_predictor_classification(train, metadata[key]["classColumn"]).to_csv('classification.csv', sep=",", index=False)

    # printDataframes(metadata[key])
    # printFolds(metadata[key])