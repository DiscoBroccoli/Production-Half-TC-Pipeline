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
            self.label = self._remove_fl(path)
            self.label = self.label.to_excel(f'{Re}_P_label_case.xlsx', index=False)

    def _remove_fl(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path)
        df = df[df['y/d'] > 0]
        df = df[df['y/d'] < 1]
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
# excel_D = ["180_D_features.xlsx", "285_D_features.xlsx", "395_D_features.xlsx", "450_D_features.xlsx", "590_D_features.xlsx"]
excel_P = ["180_P_features_case.xlsx", "285_P_features_case.xlsx", "P_features_inter_case.xlsx", "450_P_features_case.xlsx", "590_P_features_case.xlsx"]
# excel_DL = ["180_D_label.xlsx", "285_D_label.xlsx", "395_D_label.xlsx", "450_D_label.xlsx", "590_D_label.xlsx"]
excel_PL = ["180_P_label_case.xlsx", "285_P_label_case.xlsx", "PL_label_inter_case.xlsx", "450_P_label_case.xlsx", "590_P_label_case.xlsx"]

# read them in
# excels_D = [pd.ExcelFile(name) for name in excel_D]
excels_P = [pd.ExcelFile(name) for name in excel_P]
# excels_DL = [pd.ExcelFile(name) for name in excel_DL]
excels_PL = [pd.ExcelFile(name) for name in excel_PL]

# turn them into dataframes
# frames_D = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_D]
frames_P = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_P]
# frames_DL = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_DL]
frames_PL = [x.parse(x.sheet_names[0], header=None,index_col=None) for x in excels_PL]

# delete the first row for all frames except the first
# i.e. remove the header row -- assumes it's the first
# frames_D[1:] = [df[1:] for df in frames_D[1:]]
frames_P[1:] = [df[1:] for df in frames_P[1:]]
# frames_DL[1:] = [df[1:] for df in frames_DL[1:]]
frames_PL[1:] = [df[1:] for df in frames_PL[1:]]

# concatenate them..
# combined_D = pd.concat(frames_D)
combined_P = pd.concat(frames_P)
# combined_DL = pd.concat(frames_DL)
combined_PL = pd.concat(frames_PL)

# write it out
# combined_D.to_excel("D_features.xlsx", header=False, index=False)
combined_P.to_excel("P_features_case.xlsx", header=False, index=False)
# combined_DL.to_excel("DL_label.xlsx", header=False, index=False)
combined_PL.to_excel("PL_label_case.xlsx", header=False, index=False)


