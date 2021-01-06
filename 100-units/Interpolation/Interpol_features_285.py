from scipy.interpolate import interp1d
import numpy as np
from datasetTC import DATASET_DIR
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate


class ProductionDataset:
    def __init__(self):
        path = DATASET_DIR / '285_P_features.xlsx'
        self.X_feature = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        return df


P = ProductionDataset()
dicts = {}

x_i = np.linspace(0, 2, 10000)

dataframe = P.X_feature
dataframe = dataframe.drop(['y/d'], axis=1)

# creating dictionnary
for i in dataframe:
    fspline = interp1d(P.X_feature['y/d'], P.X_feature[i])
    dicts[i]=fspline(x_i)

dicts['y/d_i'] = x_i

# converting the dict to a dataframe
output_df = pd.DataFrame(data=dicts).sort_values(by=['y/d_i'])
output_df.to_excel(r'.\interpolated_f_285.xlsx', index = False)