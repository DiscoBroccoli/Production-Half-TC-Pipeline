import numpy as np
import pandas as pd
import sys
import io
print(sys.executable)
print(sys.version)

IF_180 = io.open("Interpol_features_180.py", "r")
IF_285 = io.open("Interpol_features_285.py", "r")
IF_395 = io.open("Interpol_features_395.py", "r")
IF_450 = io.open("Interpol_features_450.py", "r")
IF_590 = io.open("Interpol_features_590.py", "r")
IL_180 = io.open("Interpol_label_180.py", "r")
IL_285 = io.open("Interpol_label_285.py", "r")
IL_395 = io.open("Interpol_label_395.py", "r")
IL_450 = io.open("Interpol_label_450.py", "r")
IL_590 = io.open("Interpol_label_590.py", "r")

exec(IF_180.read())
exec(IF_285.read())
exec(IF_395.read())
exec(IF_450.read())
exec(IF_590.read())
exec(IL_180.read())
exec(IL_285.read())
exec(IL_395.read())
exec(IL_450.read())
exec(IL_590.read())
