import pandas as pd
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
        "dataframe": None,
        "columnNames": ["Sex", "Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight", "Rings"]
    },
    "breast-cancer": {
        "path": "datasets/breast-cancer/breast-cancer.data",
        "columns": [ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.NOMINAL],
        "dataframe": None,
        "columnNames": ["Sample code number", "Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion", "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses", "Class"]
    },
    "car": {
        "path": "datasets/car/car.data",
        "columns": [ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.NOMINAL],
        "ordinal": {
            0: {
                "low": 0,
                "medium": 1,
                "high": 2,
                "vhigh": 3
            },
            1: {
                "low": 0,
                "medium": 1,
                "high": 2,
                "vhigh": 3
            },
            2: {
                "2": 0,
                "3": 1,
                "4": 2,
                "5more": 3
            },
            3: {
                "2": 0,
                "4": 1,
                "more": 2
            },
            4: {
                "small": 0,
                "med": 1,
                "big": 2
            },
            5: {
                "low": 0,
                "med": 1,
                "high": 2
            }
        },
        "dataframe": None,
        "columnNames": ["buying", "maint", "doors", "persons", "lug_boot", "safety", "class"]
    },
    "forest-fires": {
        "path": "datasets/forest-fires/forest-fires.data",
        "columns": [ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.ORDINAL, ColumnTypes.ORDINAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
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
        "columnNames": ["X", "Y", "month", "day", "FFMC", "DMC", "DC", "ISI", "temp", "RH", "wind", "rain", "area"]
    },
    "house-votes": {
        "path": "datasets/house-votes/house-votes.data",
        "columns": [ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL, ColumnTypes.NOMINAL],
        "dataframe": None,
        "columnNames": ["Class Name", "handicapped-infants", "water-project-cost-sharing", "adoption-of-the-budget-resolution", "physician-fee-freeze", "el-salvador-aid", "religious-groups-in-schools", "anti-satellite-test-ban", "aid-to-nicaraguan-contras", "mx-missile", "immigration", "synfuels-corporation-cutback", "education-spending", "superfund-right-to-sue", "crime", "duty-free-exports", "export-administration-act-south-africa"]
    },
    "machine": {
        "path": "datasets/machine/machine.data",
        "columns": [ColumnTypes.NONE, ColumnTypes.NONE, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL, ColumnTypes.REAL],
        "dataframe": None,
        "columnNames": []
    }
}
    
def loadData(metadata):
    df = pd.read_table(metadata["path"], sep=',', header=None, skiprows=1)
    df.columns = metadata["columnNames"]

    return df

def handleMissingValues(dataframe: pd.DataFrame):
    return dataframe.replace('?', float("nan"))

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