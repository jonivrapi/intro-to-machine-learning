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

    print(f'stratified: {stratified(train, metadata[key]["classColumn"], 10)}')

    # metadata[key]["folds"] = partition(metadata[key])
    # print(f'training set length: {len(train)}')
    # print(f'folds: {stratified_folds(train, metadata[key]["classColumn"], 10)}')
    # print(f'prediction: {}')
    # if key == 'forest-fires':
    #     # null_model_predictor_regression(train, metadata[key]["regressionColumn"]).to_csv(f'{key}_regression.csv', sep=",", index=False)
    #     set = null_model_predictor_regression(train, metadata[key]["regressionColumn"])
    #     regressionColumn = metadata[key]["regressionColumn"]
    #     print(f'MSE: {mean_squared_error(train[regressionColumn], set["prediction"])}')
    # else:
    #     # null_model_predictor_classification(train, metadata[key]["classColumn"]).to_csv(f'{key}_classification.csv', sep=",", index=False)
    #     set = null_model_predictor_classification(train, metadata[key]["classColumn"])
    #     classColumn = metadata[key]["classColumn"]
    #     print(f'Accuracy Score: {accuracy_score(train[classColumn], set["prediction"])}')

    # # printDataframes(metadata[key])
    # # printFolds(metadata[key])

    #     classColumn = metadata[key]["classColumn"]
    #     foldsCache = stratified_folds(train, metadata[key]["classColumn"], 10)
    #     print(f'length of folds: {len(foldsCache)}')
    #     accuracy_scores = []

    #     for index, fold in enumerate(foldsCache):
    #         print(f'fold: {fold}')
    #         folds = foldsCache.copy()
    #         validation_set = folds[index]
    #         folds.pop(index)
    #         test_set = folds
    #         print(f'length of test_set: {len(test_set)}')

    #         for set in test_set:
    #             results = null_model_predictor_classification(set, classColumn)
    #             accuracyScore = accuracy_score(results[classColumn], set["prediction"])
    #             print(f'Accuracy Score {index}: {accuracyScore}')
    #             accuracy_scores.append(accuracyScore)
    #     print(f'Avg accuracy: {pd.Series(accuracy_scores).mean()}')
        