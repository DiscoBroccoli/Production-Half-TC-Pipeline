import pandas as pd
from datasetTC import INTERPOL_DIR


class Remove_firstlast:

    def __init__(self):

        reynolds = [180, 285, 395, 450, 590]

        for Re in reynolds:
            path = INTERPOL_DIR / f'interpolated_f_{Re}.xlsx'
            self.feature = self._remove_fl(path)
            self.feature = self.feature.to_excel(f'interpolated_f_{Re}_case.xlsx', index=False)

        for Re in reynolds:
            path = INTERPOL_DIR / f'interpolated_label_{Re}.xlsx'
            self.label = self._remove_flF(path)
            self.label = self.label.to_excel(f'interpolated_label_{Re}_case.xlsx', index=False)


    def _remove_fl(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        # Correcting the dissipation on the wall to be 0
        df.loc[df['y/d_i'] <= 0] = 0
        df = df[df['y/d_i'] <= 0.8]
        return df

    def _remove_flF(self, path) -> pd.DataFrame:
        print(f'Loading following dataset: {path}.')
        df = pd.read_excel(path, engine='openpyxl')
        # Correcting the dissipation on the wall to be 0
        df.loc[df['y/d_i'] <= 0, 'Prod'] = 0
        df = df[df['y/d_i'] <= 0.8]
        # df.loc[df['y/d_i'] >= 0, 'Diss'] = df['Diss']*-1
        return df


R = Remove_firstlast()
# calling to apply the _remove_fl on each dataset
r = R.feature
rr = R.label

"""
Load, process and clean the data for the Production dataset

"""


# filenames
excel_P = ["interpolated_f_180_case.xlsx", "interpolated_f_285_case.xlsx", "interpolated_f_450_case.xlsx", "interpolated_f_590_case.xlsx"]
excel_PL = ["interpolated_label_180_case.xlsx", "interpolated_label_285_case.xlsx", "interpolated_label_450_case.xlsx", "interpolated_label_590_case.xlsx"]

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
combined_P.to_excel("P_features_inter_case.xlsx", header=False, index=False, engine='openpyxl')
combined_PL.to_excel("PL_label_inter_case.xlsx", header=False, index=False, engine='openpyxl')

import shutil
from pathlib import Path
import os.path


src = Path(__file__).parent
dst = Path(__file__).parent.parent.parent / 'data'

filename1 = "P_features_inter_case.xlsx"
filename2 = "PL_label_inter_case.xlsx"

src1 = os.path.join(src, filename1)
src2 = os.path.join(src, filename2)

dst1 = os.path.join(dst, filename1)
dst2 = os.path.join(dst, filename2)

shutil.copyfile(src=src1, dst=dst1)
shutil.copyfile(src=src2, dst=dst2)