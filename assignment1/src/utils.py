import pandas as pd
class MLPipelineSetup:
    def __init__(self) -> None:
        self.datasetPaths = [
            "../../datasets/abalone/abalone.data",
            "../../datasets/breast-cancer/breast-cancer.data",
            "../../datasets/car/car.data",
            "../../datasets/forest-fires/forest-fires.data",
            "../../datasets/house-votes/house-votes.data",
            "../../datasets/machine/machine.data"
        ]

        self.dataframes = {
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


    # def handleMissingValues():

MLPipelineSetup().populateDataframes()