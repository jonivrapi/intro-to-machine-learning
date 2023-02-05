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
        "columnNames": ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    },
    # "breast-cancer": {
    #     "path": "datasets/breast-cancer/breast-cancer.data",
    #     "columns": [ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.NOMINAL],
    #     "columnTypes": [str, int, int, int, int, int, int, int, int, int, int],
    #     "dataframe": None,
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
    #     "columnNames": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    # },
    # "forest-fires": {
    #     "path": "datasets/forest-fires/forest-fires.data",
    #     "columns": [ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
    #     "columnTypes": [int, int, str, str, float, float, float, float, float, float, float, float, float],
    #     "ordinal": {
    #         2: {
    #             "jan": 0,
    #             "feb": 1,
    #             "mar": 2,
    #             "apr": 3,
    #             "may": 4,
    #             "jun": 5,
    #             "jul": 6,
    #             "aug": 7,
    #             "sep": 8,
    #             "oct": 9,
    #             "nov": 10,
    #             "dec": 11
    #         },
    #         3: {
    #             "mon": 0,
    #             "tue": 1,
    #             "wed": 2,
    #             "thu": 3,
    #             "fri": 4,
    #             "sat": 5,
    #             "sun": 6
    #         }
    #     },
    #     "dataframe": None,
    #     "columnNames": ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]
    # },
    # "house-votes": {
    #     "path": "datasets/house-votes/house-votes.data",
    #     "columns": [ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL],
    #     "columnTypes": [str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str, str],
    #     "dataframe": None,
    #     "columnNames": ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
    # },
    # "machine": {
    #     "path": "datasets/machine/machine.data",
    #     "columns": [ColumnTypes.NONE, ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
    #     "columnTypes": [str, str, int, int, int, int, int, int, int, int],
    #     "dataframe": None,
    #     "columnNames": ["vendor name", "Model Name", "MYCT", "MMIN", "MMAX", "CACH", "CHMIN", "CHMAX", "PRP", "ERP"]
    # }
}
    
def loadData(metadata):
    df = pd.read_table(metadata["path"], sep=',', header=None, skiprows=1)
    df.columns = metadata["columnNames"]

    return df

def handleMissingValues(metadata):
    df: pd.DataFrame = metadata["dataframe"]
    df.replace('?', np.nan, inplace=True)
    for index, coltype in enumerate(metadata["columnTypes"]):
        colName = metadata["columnNames"][index]
        # print(f'colname: {colName}')
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
    # print(f'in metadata: {metadata["dataframe"]}')
    for index, item in enumerate(metadata["columns"]):
        colName = metadata["columnNames"][index]
        # print(f'index: {index}')
        if item == ColumnTypes.NOMINAL:
            oneHotEncoding = pd.get_dummies(metadata["dataframe"][colName])
            # print(f'one hot: {oneHotEncoding}')
            metadata["dataframe"].drop(metadata["columnNames"][index], axis=1, inplace=True)
            metadata["dataframe"] = metadata["dataframe"].join(oneHotEncoding, rsuffix=f"_{colName}")
    
    return metadata["dataframe"]

def discretization(metadata, equalwidth=True, numBins=4, numQuantiles=4):
    if equalwidth:
        for index, item in enumerate(metadata["columns"]):
            colName = metadata["columnNames"][index]
            if item == ColumnTypes.REAL:
                # print(f'df: {metadata["dataframe"][colName]}')
                metadata["dataframe"][f"{colName}_discretized"] = pd.cut(metadata["dataframe"][colName], bins=numBins)
    else:
        for index, item in enumerate(metadata["columns"]):
            colName = metadata["columnNames"][index]
            if item == ColumnTypes.REAL:
                # print(f'df: {metadata["dataframe"][colName]}')
                metadata["dataframe"][f"{colName}_discretized"] = pd.qcut(metadata["dataframe"][colName], q=numQuantiles, duplicates='drop')
    
    return metadata["dataframe"]

# def zScoreStandardization(metadata, trainingSet: pd.DataFrame, testSet: pd.DataFrame)

def randomPartition(metadata):
    df = metadata["dataframe"]

    msk = np.random.rand(len(df)) < 0.8

    train = df[msk]

    test = df[~msk]

    # print(f'dataframe length: {len(df)}')
    # print(f'80%: {len(df) * 0.8} | 20%: {len(df) * 0.2}')
    # print(f'train: {train}')
    # print(f'test: {test}')

    return train, test

def zScore(train: pd.DataFrame, test: pd.DataFrame, metadata):
    for index, coltype in enumerate(metadata["columnTypes"]):
        colName = metadata["columnNames"][index]
        if coltype == int or coltype == float:
            print(f'col: {train.loc[:, colName]}')
            mu = train.loc[:, colName].mean()
            print(f'mu: {mu}')

        

# def oneHotEncode():
    

# def populateDataframes(self):
#     for index, key in enumerate(self.dataframes):
#         self.dataframes[key] = self.loadData(index)
    
#     return self



def printDataframes(dataframe) -> None:
    print(dataframe)

def getCurrentWorkdingDirectory() -> None:
    import os
    print("Working Directory: {}".format(os.getcwd()))