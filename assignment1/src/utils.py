import pandas as pd
class MLPipelineBuilder:
    def __init__(self) -> None:
        self.datasetPaths: list[str] = [
            "../../datasets/abalone/abalone.data",
            "../../datasets/breast-cancer/breast-cancer.data",
            "../../datasets/car/car.data",
            "../../datasets/forest-fires/forest-fires.data",
            "../../datasets/house-votes/house-votes.data",
            "../../datasets/machine/machine.data"
        ]

        self.dataframes: dict[str, pd.DataFrame] = {
            "abalone": None,
            "breast-cancer": None,
            "car": None,
            "forest-fires": None,
            "house-votes": None,
            "machine": None
        }
    
    def loadData(self, index):
        return pd.read_table(self.datasetPaths[index], sep=',', header=None)
    

    def populateDataframes(self):
        for index, key in enumerate(self.dataframes):
            self.dataframes[key] = self.loadData(index)
        
        return self


    def handleMissingValues(self):
        for key in self.dataframes:
            self.dataframes[key].replace('?', float("nan"), inplace=True)

        return self

    def printDataframes(self) -> None:
        print(self.dataframes)


(MLPipelineBuilder()
    .populateDataframes()
    .handleMissingValues()
    .printDataframes())