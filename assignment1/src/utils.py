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
    
    def loadData(path):
        return pd.read_table(path, sep=',', header=None)
    
    # def handleMissingValues():


for path in datasetPaths:
    print(loadData(path))