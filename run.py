""" OFET 3.0 """

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
import ofet_



L_ofets = []
L_list = [80, 60 ,50]


W = 1.3
C = 50
folder_data_path = 'Curvas_de_Saida_e_Transferencia/'
set_folder = 'parameters_L'


for L in L_list:
    geos = {'values': [W,L,C] , 'se_values' : [1e-3, 1e-6 , 1e-10]}
    relative_path = f"{L}um/"
    print('Analize the ofet on the folder : ', relative_path)
    x_ofet = ofet_.ofet(geos)
    x_ofet.analizy_ofet(L , folder_data_path ,relative_path , set_folder, export_parameters = False, only_graphics=False)
    L_ofets.append(x_ofet)



GroupL = ofet_.CompactGroupTOfets(L_ofets, set_folder,L_list)
GroupL.run_analysis()
