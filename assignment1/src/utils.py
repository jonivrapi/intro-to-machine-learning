import pandas as pd
import numpy as np
from enum import Enum

class ColumnTypes(Enum):
    ORDINAL = 1,
    NOMINAL = 2,
    REAL = 3,
    NONE = 1

metadata: dict[str, dict] = {
    "abalone": {
        "path": "datasets/abalone/abalone.data",
        "columns": [ColumnTypes.NOMINAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
        "columnTypes": [str, float, float, float, float, float, float, float, int],
        "dataframe": None,
        "folds": None,
        "classColumn": "Diameter_discretized",
        "columnNames": ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    },
    # "breast-cancer": {
    #     "path": "datasets/breast-cancer/breast-cancer.data",
    #     "columns": [ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.NOMINAL],
    #     "columnTypes": [str, int, int, int, int, int, int, int, int, int, int],
    #     "dataframe": None,
    #     "folds": None,
    #     "classColumn": "Class",
    #     "columnNames": ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    # },
    # "car": {
    #     "path": "datasets/car/car.data",
    #     "columns": [ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.NOMINAL],
    #     "columnTypes": [str, str, str, str, str, str, str],
    #     "ordinal": {
    #         0: {
    #             "low": 0,
    #             "medium": 1,
    #             "high": 2,
    #             "vhigh": 3
    #         },
    #         1: {
    #             "low": 0,
    #             "medium": 1,
    #             "high": 2,
    #             "vhigh": 3
    #         },
    #         2: {
    #             "2": 0,
    #             "3": 1,
    #             "4": 2,
    #             "5more": 3
    #         },
    #         3: {
    #             "2": 0,
    #             "4": 1,
    #             "more": 2
    #         },
    #         4: {
    #             "small": 0,
    #             "med": 1,
    #             "big": 2
    #         },
    #         5: {
    #             "low": 0,
    #             "med": 1,
    #             "high": 2
    #         }
    #     },
    #     "dataframe": None,
    #     "folds": None,
    #     "classColumn": "class",
    #     "columnNames": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    # },
    "forest-fires": {
        "path": "datasets/forest-fires/forest-fires.data",
        "columns": [ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
        "columnTypes": [int, int, str, str, float, float, float, float, float, float, float, float, float],
        "ordinal": {
            2: {
                "jan": 0,
                "feb": 1,
                "mar": 2,
                "apr": 3,
                "may": 4,
                "jun": 5,
                "jul": 6,
                "aug": 7,
                "sep": 8,
                "oct": 9,
                "nov": 10,
                "dec": 11
            },
            3: {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6
            }
        },
        "dataframe": None,
        "regressionColumn": "ISI",
        "columnNames": ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]
    },
    # "house-votes": {
    #     "path": "datasets/house-votes/house-votes.data",
    #     "columns": [ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL],
    #     "columnTypes": [str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str],
    #     "dataframe": None,
    #     "folds": None,
    #     "classColumn": "Class Name",
    #     "columnNames": ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
    # },
    # "machine": {
    #     "path": "datasets/machine/machine.data",
    #     "columns": [ColumnTypes.NONE, ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
    #     "columnTypes": [str, str, int, int, int, int, int, int, int, int],
    #     "dataframe": None,
    #     "folds": None,
    #     "classColumn": "PRP",
    #     "columnNames": ["vendor name", "Model Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    # }
}
    
def loadData(metadata):
    #fix header skipping
    df = pd.read_table(metadata["path"], sep=',', header=None, skiprows=1)
    df.columns = metadata["columnNames"]

    return df

def handleMissingValues(metadata):
    #congressional data set is not missing values - do dummies on yes/no/? for that
    df: pd.DataFrame = metadata["dataframe"]
    df.replace('?', np.nan, inplace=True)
    for index, coltype in enumerate(metadata["columnTypes"]):
        colName = metadata["columnNames"][index]
        if coltype == int:
            df[colName] = pd.to_numeric(df[colName])
            df[colName] = df[colName].fillna(df.loc[:, colName].mean().astype(int))
        if coltype == float:
            df[colName] = df[colName].fillna(df.loc[:, colName].mean().astype(float))

    return df

def encodeOrdinalFeatures(metadata):
    if "ordinal" in metadata:
        for key in metadata["ordinal"]:
            metadata["dataframe"].iloc[:, key].replace(metadata["ordinal"][key], inplace=True)
    
    return metadata["dataframe"]

def encodeNominalFeatures(metadata):
    colsToDrop = []
    indicesToDrop = []

    for index, item in enumerate(metadata["columns"]):
        colName = metadata["columnNames"][index]
        if item == ColumnTypes.NOMINAL:
            oneHotEncoding = pd.get_dummies(metadata["dataframe"][colName])
            colsToDrop.append(colName)
            indicesToDrop.append(index)
            # metadata["dataframe"].drop(colName, axis=1, inplace=True)
            metadata["dataframe"] = metadata["dataframe"].join(oneHotEncoding, rsuffix=f"_{colName}")
    
    # for colName in colsToDrop:
    #     metadata["columnNames"] = [x for x in metadata["columnNames"] if x != colName]

    # for index in indicesToDrop:
    #     metadata["columns"] = metadata["columns"][:index] + metadata["columns"][index+1:]
    #     metadata["columnTypes"] = metadata["columnTypes"][:index] + metadata["columnTypes"][index+1:]

    
    return metadata["dataframe"]

def discretization(metadata, equalwidth=True, numBins=4, numQuantiles=4):
    if equalwidth:
        for index, item in enumerate(metadata["columnTypes"]):
            colName = metadata["columnNames"][index]
            if item == float:
                metadata["dataframe"][f"{colName}_discretized"] = pd.cut(metadata["dataframe"][colName], bins=numBins)
    else:
        for index, item in enumerate(metadata["columnTypes"]):
            if item == float:
                colName = metadata["columnNames"][index]
                metadata["dataframe"][f"{colName}_discretized"] = pd.qcut(metadata["dataframe"][colName], q=numQuantiles, duplicates='drop')
    
    return metadata["dataframe"]

def zScore(train: pd.DataFrame, test: pd.DataFrame, metadata):
    for index, coltype in enumerate(metadata["columnTypes"]):
        if coltype == int or coltype == float:
            colName = metadata["columnNames"][index]
            mu = train[colName].mean()
            sigma = train[colName].std()
            z_scores = (train[colName] - mu) / sigma
            #could drop the original column
            train = train.assign(**{f"{colName}_z_score": z_scores})
            test = test.assign(**{f"{colName}_z_score": z_scores})

    return train, test   

def randomPartition(metadata):
    df = metadata["dataframe"]

    msk = np.random.rand(len(df)) < 0.8

    train = df[msk]

    test = df[~msk]

    # print(f'train: {train}')
    # print(f'test: {test}')

    return train, test

def partition(metadata):
    print(f'cols: {stratify(metadata["dataframe"], metadata["classColumn"])}')
    return createFolds(metadata["dataframe"], numfolds=10)


def stratify(df, targetColumn):
    # first randomize entire dataset
    # split in half
    # count instances of each class, for each one of these
    # 
    classCounts = df[targetColumn].value_counts()
    classRatios = classCounts / classCounts.sum()
    df_stratified = pd.DataFrame(columns=df.columns)
    for target, ratio in classRatios.items():
        target_df = df[df[targetColumn] == target]
        #this has an issue where i could possibly pull the same rows twice, use frac=1 to return randomized total dataset
        target_df_sampled = target_df.sample(frac=ratio, random_state=1)
        df_stratified = pd.concat([df_stratified, target_df_sampled])
    return df_stratified

def createFolds(df, numfolds=2):
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)
    fold_size = int(df.shape[0] / numfolds)
    folds = []
    for i in range(numfolds):
        start_index = i * fold_size
        end_index = (i + 1) * fold_size
        fold = df.iloc[start_index:end_index]
        folds.append(fold)
    # print(f"folds: {folds}")
    return folds

def stratified_folds(df, target_column, n):
    # Group the dataframe by the target column
    grouped = df.groupby(target_column)
    # Get the count of the classes
    class_counts = grouped.size().reset_index(name='counts')
    # Calculate the average number of samples per fold
    avg_samples = int(df.shape[0]/n)
    # Create an empty list to store the folds
    folds = []
    for i in range(n):
        fold = []
        for index, row in class_counts.iterrows():
            class_data = grouped.get_group(row[target_column])
            samples = min(avg_samples, row['counts'])
            fold.append(class_data.sample(samples, random_state=i))
        folds.append(pd.concat(fold, axis=0))
    return folds



def printDataframes(dataframe) -> None:
    print(dataframe["dataframe"])

def printFolds(dataframe) -> None:
    print(dataframe["folds"])

def getCurrentWorkdingDirectory() -> None:
    import os
    print("Working Directory: {}".format(os.getcwd()))

def accuracy_score(truth, prediction):
    # print(f'truth: {truth}')
    # print(f'prediction: {prediction}')
    
    y_true = pd.Series(truth)
    y_pred = pd.Series(prediction)
    return (y_true == y_pred).mean()

def mean_squared_error(truth, prediction):
    # print(f'truth: {truth}')
    # print(f'prediction: {prediction}')
    
    y_true = pd.Series(truth)
    y_pred = pd.Series(prediction)

    return ((y_true - y_pred) ** 2).mean()



def stratify_k_fold(df, column, k):
  # Group the dataframe by the class column
  grouped = df.groupby(column)
  # Get the count of the classes
  class_counts = grouped.size().reset_index(name='counts')
  # Calculate the average number of samples per fold
  avg_samples = int(df.shape[0]/k)
  # Create an empty list to store the folds
  folds = []
  for i in range(k):
    fold = []
    for index, row in class_counts.iterrows():
      class_data = grouped.get_group(row[column])
      samples = min(avg_samples, row['counts'])
      fold.append(class_data.sample(samples, random_state=i))
    folds.append(pd.concat(fold, axis=0))
  return folds

def null_model_predictor_classification(df, target_column):
    # Count the number of unique classes in the target column
    classes = df[target_column].value_counts()
    # Calculate the frequency of each class in the target column
    class_frequencies = classes/df.shape[0]
    # Create a dictionary with the class as the key and the frequency as the value
    class_frequency_dict = class_frequencies.to_dict()
    # Define a function that will predict the most frequent class for all samples
    def predict(x):
        return max(class_frequency_dict, key=class_frequency_dict.get)
    # Apply the predict function to the target column to generate predictions
    df['prediction'] = df.apply(predict, axis=1)

    return df

def null_model_predictor_regression(df, target_column):
    # Calculate the mean of the target column
    mean = df[target_column].mean()
    # Define a function that will predict the mean for all samples
    def predict(x):
        return mean
    # Apply the predict function to the target column to generate predictions
    df['prediction'] = df.apply(predict, axis=1)
    return df

