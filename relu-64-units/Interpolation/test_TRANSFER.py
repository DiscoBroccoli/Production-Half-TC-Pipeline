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