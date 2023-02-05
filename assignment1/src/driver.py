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
    printDataframes(metadata[key]["dataframe"])

