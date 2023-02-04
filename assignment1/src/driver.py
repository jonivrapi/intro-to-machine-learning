from utils import *

# Load dataframes into metadata map
for key in metadata:
    print(f'key: {key}')
    
    metadata[key]["dataframe"] = loadData(metadata[key])
    metadata[key]["dataframe"] = handleMissingValues(metadata[key]["dataframe"])
    metadata[key]["dataframe"] = encodeOrdinalFeatures(metadata[key])
    metadata[key]["dataframe"] = encodeNominalFeatures(metadata[key])
    metadata[key]["dataframe"] = discretization(metadata[key], True)
    
    printDataframes(metadata[key]["dataframe"])

