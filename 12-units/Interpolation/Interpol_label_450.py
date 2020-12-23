from scipy.interpolate import interp1d
import numpy as np
from datasetTC import DATASET_DIR
import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy import interpolate


class ProductionDataset:
    def __init__(self):
        path = DATASET_DIR / '450_P_label.xlsx'
        self.X_test = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        return df


P = ProductionDataset()
dicts = {}

x_i = np.linspace(0, 2, 10000)

dataframe = P.X_test
dataframe = dataframe.drop(['y/d'], axis=1)

# creating dictionnary
for i in dataframe:
    fspline = interp1d(P.X_test['y/d'], P.X_test[i])
    dicts[i] = fspline(x_i)

dicts['y/d_i'] = x_i

# Need to re-order the column to match the CFD dataframe
# desired order
key_order = ['y/d_i', 'Prod']
# use collections method OrderedDict
dicts = OrderedDict(dicts)
for k in key_order: # a loop to force the order you want
    dicts.move_to_end(k)

# converting the dict to a dataframe
# re-arrange the column because of random state in train_test_split
output_df = pd.DataFrame(data=dicts).sort_values(by=['y/d_i'])
output_df.to_excel(r'.\interpolated_label_450.xlsx', index = False)

plt.title('Production Interpolation Validation- $u_{\\tau} = 450 $')

fig = plt.gcf()
plt.figure
plt.plot(P.X_test['y/d'],dataframe, '-')
plt.plot(dicts['y/d_i'],dicts['Prod'], '--')
plt.legend(['data', 'interpolated'], loc='best')
# plt.show()
fig.savefig(r'./Image/label_450.png', dpi=500)
'''
class ProductionDataset:
    def __init__(self):
        path = DATASET_DIR / '285_P_label.xlsx'
        self.X_test = self._load_dataset(path)

    def _load_dataset(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path)
        return df


P = ProductionDataset()

x = P.X_test['yplus']
y = P.X_test['Prod']
f = interp1d(x, y)
f2 = interp1d(x, y, kind='cubic')

xnew = np.linspace(0, 569, num=41, endpoint=True)
import matplotlib.pyplot as plt
plt.plot(x, y, 'o', xnew, f(xnew), '-', xnew, f2(xnew), '--')
plt.legend(['data', 'linear', 'cubic'], loc='best')
plt.show()
'''




