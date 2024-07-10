
""" OFET 3.0 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import ofet_


W = 1.3
C = 50

T_ofets = []
T_list = [288 , 290, 292, 294,299,304,309]
geos = {'values': [W,60,C] , 'se_values' : [1e-3, 1e-6 , 1e-10] }

print(np.array(geos['values']) * np.array(geos['se_values']) )
folder_data_path = '60um_temperatura/' 
set_folder = 'parameters_T'

for T in T_list:
    relative_path = f"{T}K_He/"
    print(relative_path)
    x_ofet = ofet_.ofet(geos, T = T)
    x_ofet.analizy_ofet(T , folder_data_path ,relative_path , set_folder, export_parameters = False, only_graphics=False)
    T_ofets.append(x_ofet)

GroupT = ofet_.CompactGroupTOfets(T_ofets, set_folder,T_list)
GroupT.run_analysis()
















