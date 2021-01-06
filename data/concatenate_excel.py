import pandas as pd
#from datasetTC import DATASET_DIR, INTERPOL_DIR


class Remove_firstlast:
    
    def __init__(self):
        
        reynolds = [180, 285, 395, 450, 590]

        for Re in reynolds:
            path = f'{Re}_P_features.xlsx'
            self.feature = self._remove_fl(path)
            self.feature = self.feature.to_excel(f'{Re}_P_features_case.xlsx', index=False)

        for Re in reynolds:
            path = f'{Re}_P_label.xlsx'
            self.label = self._remove_flF(path)
            self.label = self.label.to_excel(f'{Re}_P_label_case.xlsx', index=False)

    def _remove_fl(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        # Correcting the dissipation on the wall to be 0
        df.loc[df['y/d'] <= 0] = 0
        df = df[df['y/d'] <= 0.8]
        return df

    def _remove_flF(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        # Correcting the dissipation on the wall to be 0
        df.loc[df['y/d'] <= 0, 'Prod'] = 0
        df = df[df['y/d'] <= 0.8]
        return df


R = Remove_firstlast()
# calling to apply the _remove_fl on each dataset
r = R.feature
rr = R.label

"""
Load, process and clean the data for the Production dataset

"""


import pandas as pd

# filenames
excel_P = ["180_P_features_case.xlsx", "285_P_features_case.xlsx", "P_features_inter_case.xlsx", "450_P_features_case.xlsx", "590_P_features_case.xlsx"]
excel_PL = ["180_P_label_case.xlsx", "285_P_label_case.xlsx", "PL_label_inter_case.xlsx", "450_P_label_case.xlsx", "590_P_label_case.xlsx"]

# read them in
excels_P = [pd.ExcelFile(name) for name in excel_P]
excels_PL = [pd.ExcelFile(name) for name in excel_PL]

# turn them into dataframes
frames_P = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_P]
frames_PL = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_PL]

# delete the first row for all frames except the first
# i.e. remove the header row -- assumes it's the first
frames_P[1:] = [df[1:] for df in frames_P[1:]]
frames_PL[1:] = [df[1:] for df in frames_PL[1:]]

# concatenate them..
combined_P = pd.concat(frames_P)
combined_PL = pd.concat(frames_PL)

# write it out
combined_P.to_excel("P_features_case.xlsx", header=False, index=False)
combined_PL.to_excel("PL_label_case.xlsx", header=False, index=False)


