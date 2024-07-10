""" OFET 3.0 """

"""This module create an Ofet and set_ofet objects to analyse the electrical, thermal and geometry properties on organic field effect transistors - OFETs. It's the 3.0 versión using POO and sklearn for the machine learning on the linar and cuadratic regresions."""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math 

import os
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def determinate_linear_regresion(x, y):
    model = LinearRegression()

    model.fit(x,y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x, y)
    return [slope , intercept , r_squared]

def determine_quadratic_regression(x, y):
    # Crear una matriz de características cuadráticas
    X = np.column_stack((x, x**2))

    # Crear un modelo de regresión lineal
    modelo = LinearRegression()

    # Ajustar el modelo a los datos
    modelo.fit(X, y)

    # Obtener los coeficientes para la regresión cuadrática y lineal
    b = modelo.coef_[0]  # Coeficiente lineal
    a = modelo.coef_[1]  # Coeficiente cuadrático
    c = modelo.intercept_  # Término indeslope

    # Calcular el coeficiente de determinación (R^2)
    r_cuadrado = modelo.score(X, y)

    return [a, b, c, r_cuadrado]

from scipy.stats import linregress

def determinate_linear_regresion_2(x, y):
    # Realiza la regresión lineal
    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    # Calcula la desviación estándar de los residuos
    residuals = y - (slope * x + intercept)
    std_residuals = np.std(residuals)

    # Calcula la desviación estándar de la slope y el intercept
    std_slope = std_residuals / np.sqrt(np.sum((x - np.mean(x))**2))
    std_intercept = std_residuals * np.sqrt(1/len(x) + np.mean(x)**2 / np.sum((x - np.mean(x))**2))

    # Calcula el error estándar de la slope y el intercept
    se_slope = std_slope / np.sqrt(len(x))
    se_intercept = std_intercept / np.sqrt(len(x))

    # Devuelve los resultados
    resultados = {
        'slope': slope,
        'intercept': intercept,
        'std_slope': std_slope,
        'std_intercept': std_intercept,
        'se_slope': se_slope,
        'se_intercept': se_intercept,
        'r_value' : r_value
    }
    return resultados

import cmath  # Required for handling square roots of negative numbers

def calculate_roots(parameters):
    a, b, c = parameters
    # Calculate the discriminant
    discriminant = cmath.sqrt(b**2 - 4*a*c)

    # Calculate the two roots
    root1 = (-b + discriminant) / (2*a)
    root2 = (-b - discriminant) / (2*a)

    return [root1, root2]

def calculate_vertice(parameters):
    a, b, c = parameters
    # Coordenada x del vértice
    h = -b / (2 * a)
    # Coordenada y del vértice
    k = a * h**2 + b * h + c
    return [h, k]

def calcular_incertidumbre_operacion(op, valores, incertidumbres):
    # Asegurarse de que la operación y las listas tengan la misma longitud
    if len(valores) != len(incertidumbres):
        raise ValueError("Las listas de valores y de incertidumbres deben tener la misma longitud.")

    # Calcular la incertidumbre de la operación
    resultado = op(*valores)
    incertidumbre_resultado = 0.0

    for i in range(len(valores)):
        # Calcular la derivada parcial respecto al i-ésimo parámetro
        derivada_parcial = np.prod([incertidumbres[j] if j != i else 1 for j in range(len(valores))])

        # Sumar cuadrados de las derivadas parciales ponderadas por las incertidumbres
        incertidumbre_resultado += (derivada_parcial * incertidumbres[i])**2

    # Tomar la raíz cuadrada del resultado
    incertidumbre_resultado = np.sqrt(incertidumbre_resultado)

    return resultado, incertidumbre_resultado


def solve_system(df, variable_name, x_name, y_name):
    # Get column values as arrays
    mu_values = df[variable_name].values
    x_values = df[x_name].values
    y_values = df[y_name].values

    # Build matrix A with cross terms between x and y and powers of x and y
    A = np.column_stack((
        np.ones_like(x_values),
        x_values,
        x_values**2,
        y_values,
        y_values**2,
        x_values * y_values,  # Cross terms between x and y
        x_values**2 * y_values,  # x**2 * y
        x_values * y_values**2,  # x * y**2
        x_values**2 * y_values**2  # x**2 * y**2
    ))

    # Solve the system of linear equations A * X = mu_values
    # Where X is a vector containing the unknown coefficients
    X, residuals, rank, s = np.linalg.lstsq(A, mu_values, rcond=None)

    # You can use these coefficients to form the original equation with subscripts for powers
    equation = (
        f"{X[0]} + {X[1]}*{x_name} + {X[2]}*{x_name}**2 + {X[3]}*{y_name} + {X[4]}*{y_name}**2 + "
        f"{X[5]}*{x_name}*{y_name} + {X[6]}*{x_name}**2*{y_name} + {X[7]}*{x_name}*{y_name}**2 + "
        f"{X[8]}*{x_name}**2*{y_name}**2"
    )

    # Additionally, X contains the coefficients A in the order [a, b, c, d, e, f, g, h, i]
    coefficients = X
    coefficients.resize((3,3))
    return equation, coefficients


class ofet:

    def __init__(self, geos , T = 298.15) -> None:
        self.geos = geos
        self.W , self.L , self.C = np.array(geos['values']) * np.array(geos['se_values'])  # geo = [W,L,C] unit W = mm
        self.T = T # K
        self.geos['_values'] = np.array([self.W , self.L , self.C])

        self.se_T = 0.5  # K
        self.se_C = geos['se_values'][2] #10**(np.log10(self.C)*0.5)
        self.geos['se_values'] = np.array(self.geos['se_values'])* 0.5

        self.dots_diff_regions = []
        self.saturation_max_current = pd.DataFrame()

        self.tables = {}
        self.graphic_folder = 'graphics/'
        self.graphics = {}
        self.histeresis = {}
        self.electric_model = []


    def calculate_mobility_factor(self):

        def operacion(W , L , C):
            return L/(W*C)
        
        valores_geo = [self.W , self.L , self.C]
        incertidumbre_geo = np.array(self.geos['se_values'])
        self.mobility_factor , self.incertidumbre_mobility_factor  = calcular_incertidumbre_operacion(operacion , valores_geo , incertidumbre_geo)

    def process_txt_files(self, root_folder):
        txt_files_dict = {}

        for root, dirs, files in os.walk(root_folder):
            current_subfolder = os.path.basename(root)
            txt_files_dict[current_subfolder] = []

            for file_name in files:
                if file_name.endswith(".txt"):
                    file_path = os.path.join(root, file_name)
                    txt_files_dict[current_subfolder].append(file_path)


        for subfolder, txt_files in list(txt_files_dict.items())[1:]:
            self.tables[subfolder] = {}

            for txt_file in txt_files:

                if 'Vg' in txt_file:
                    header = ['v_var', 'ig' , 'v_fixed' , 'ids']
                if 'Vds' in txt_file:
                    header = ['v_var', 'ids' , 'v_fixed' , 'ig']

                df = pd.read_csv(txt_file,sep = '\t')
                df.columns = header # O la función adecuada para cargar los datos en un DataFrame
                self.tables[subfolder][txt_file] = df
        return self.tables


    def calculate_derivative(self, path_folder, path,  x_col, y_col, derivative_col_name):
        x = self.tables[path_folder][path][x_col].values
        y = self.tables[path_folder][path][y_col].values

        # Desactivar las advertencias temporariamente
        old_settings = np.seterr(all='ignore')

        # Calcular la derivada numérica directamente desde y
        derivative = np.gradient(y, x)

        # Restaurar las configuraciones de advertencias originales
        np.seterr(**old_settings)

        self.tables[path_folder][path][derivative_col_name] = derivative

    def calculate_roots(self, path_folder, path,  x_col):
        x = self.tables[path_folder][path][x_col].values
        self.tables[path_folder][path][f'root_{x_col}'] = np.sqrt(abs(x))
        self.tables[path_folder][path][f'log10_{x_col}'] = np.log10(abs(x))

    def calculate_mobility(self, path_folder, path,  x_col):
        x_sat = self.tables[path_folder][path][f'derivative_root_{x_col}'].values
        x_lin = self.tables[path_folder][path][f'derivative_{x_col}'].values
        vds = self.tables[path_folder][path]['v_fixed'].values
        self.tables[path_folder][path][f'derivative_root_{x_col}**2'] = x_sat*x_sat
        self.tables[path_folder][path]['mobility_sat'] = (2*self.mobility_factor) * (x_sat*x_sat)
        # self.tables[path_folder][path]['mobility_linear'] = (self.mobility_factor/ vds) * x_lin

    def define_sweep(self, path_folder, path):
        x = self.tables[path_folder][path]['v_var'].values
        dx = np.gradient(x)
        sign_changes = np.where(np.diff(np.sign(dx)))[0] + 1
        self.tables[path_folder][path].loc[:sign_changes[0] , 'sweep'] = 'back'
        self.tables[path_folder][path].loc[sign_changes[0]: , 'sweep'] = 'forth'


    def define_region_aux(self, path_folder, path, sweep):
        condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
        x_ = self.tables[path_folder][path][condition_sweep]

        vds = round(x_['v_fixed'].values.mean(), 1)
        vg = x_['v_var'].values
        d_root_ids = x_['derivative_log_ids'].values

        condition_sigma = abs(d_root_ids) <= 3*abs(d_root_ids.std(ddof = 0))
        y = d_root_ids[condition_sigma]
        x = vg[condition_sigma]

        if len(y) > 0:
            min_index = np.where(y == min(y))[0]

            if len(min_index) > 0:
                line = [path_folder, path , sweep, vds, x[min_index[0]] , y[min_index[0]] ]
                self.dots_diff_regions.append(line)


    def define_region(self, path_folder, path, sweep):
        self.dots_diff_regions_df = pd.DataFrame( self.dots_diff_regions )
        self.dots_diff_regions_df.columns = [ 'path_folder' , 'path' , 'sweep', 'vds', 'vg_d_log_min' , 'd_log_min' ]
        path_folder_condition = self.dots_diff_regions_df['path_folder'] == path_folder
        path_condition = self.dots_diff_regions_df['path'] == path
        sweep_condition = self.dots_diff_regions_df['sweep'] == sweep

        df_min = self.dots_diff_regions_df[path_condition & path_folder_condition  & sweep_condition ]
        x_min = df_min['vg_d_log_min'].values

        if len(x_min) == 0:
            x_min = [0]

        condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
        self.tables[path_folder][path].loc[condition_sweep, 'region']= np.where(self.tables[path_folder][path][condition_sweep]['v_var'] <= x_min[0] , 'saturacion', 'linear')


    def calculate_saturation_max_current_values(self, path_folder , path, keys):
        required_columns = keys
        if not all(col in self.saturation_max_current.columns for col in required_columns):
            self.saturation_max_current = pd.DataFrame(columns=required_columns)

        table = self.tables[path_folder][path]
        x_ = table.groupby('sweep')[keys].max().reset_index()
        print(x_)


        self.saturation_max_current = pd.concat([self.saturation_max_current, x_], ignore_index=True).reset_index()


    def saturacion_curve(self, path_folder, path,  y1_col, y2_col, sweep=None, folder='graphic', axes = None):
        
        relabels = {
            'root_ids': r'$\sqrt{i_{ds}}$',
            'derivative_log_ids': r'$\frac{\partial{\log ids}}{\partial{V_{g}}}$',
            'derivative_log_ids_on': r'$I_{on}$'
        }

        os.makedirs(folder + '/saturation', exist_ok = True)

        if sweep not in ['back', 'forth', None]:
          print('There is a problem with the sweep please select on of this : back, forth, None')
          pass

        else:
            condition_sweep = self.tables[path_folder][path]['sweep'] == sweep

            table_test = self.tables[path_folder][path][condition_sweep]
            vds = round(table_test['v_fixed'].values.mean(),1)

            x1 = table_test['v_var'].values
            y1 = table_test[y1_col].values

            derivative_condition = abs(table_test['v_var']) < max(abs(table_test['v_var']))
            x2 = table_test[derivative_condition]['v_var'].values
            y2 = table_test[derivative_condition][y2_col].values

            sns.set_theme(style="ticks")

            # Plotting with Seaborn
            figure, ax1 = plt.subplots(figsize=(8,7))


            # Plot the first line on the left axis (ax1)
            sns.lineplot(x=x1, y=y1, ax=ax1, label=relabels[y1_col])

            # Create the right axis (ax2) sharing the same x-axis
            ax2 = ax1.twinx()

            # Plot the lines on the right axis (ax2)
            sns.lineplot(x=x2, y=y2, ax=ax2, color='orange', label=relabels[y2_col])


            condition_sweep = self.dots_diff_regions_df['sweep'] == sweep
            condition_vds = self.dots_diff_regions_df['vds'] == vds

            dot = self.dots_diff_regions_df[condition_sweep &  condition_vds]

            x2_dot = dot['vg_d_log_min'].values 
            y2_dot = dot['d_log_min'].values

            sns.scatterplot(x=x2_dot, y=y2_dot, ax=ax2, color='black', label=relabels[f"{y2_col}_on"])

            #
            if axes is not None:
                ax1.set_ylim(axes['ax1_min'], axes['ax1_max'])
                ax2.set_ylim(axes['ax2_min'], axes['ax2_max'])

            # Adjust legends to be at the center
            ax1.legend(loc='upper left', bbox_to_anchor=(0, 1.15))
            ax2.legend(loc='upper right', bbox_to_anchor=(1, 1.15) , ncol = 2)
            
            ax1.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
            ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

            # Add a title to the plot
            plt.title( f"Saturacion {sweep} vds = {str(abs(vds))} V" )

            # Save the plot
            plt.savefig(f"{folder}saturation/saturation_{sweep}_{str(abs(vds))}.png")
            plt.close(figure)


    def calculate_log_derivative(self, path_folder, path, x_col):
        x = self.tables[path_folder][path][x_col].values
        self.tables[path_folder][path][f"log_{x_col}"] = np.log10(abs(x))
       

    def create_caracteristics(self, col_name = 'ids'):
        self.calculate_mobility_factor()

        for path_folder in ['Vds' , 'Vg']:
            for path in self.tables[path_folder]:
                self.define_sweep(path_folder, path)
                self.calculate_roots(path_folder, path, col_name)

                if path_folder == 'Vds':
                    self.calculate_derivative(path_folder, path, 'v_var', col_name, f'd_{col_name}/d_{path_folder}')


                if path_folder == 'Vg':
                    self.calculate_derivative(path_folder, path, 'v_var', col_name, f'derivative_{col_name}')
                    self.calculate_derivative(path_folder, path, 'v_var', f'derivative_{col_name}', f'second_derivative_{col_name}')
                    self.calculate_derivative(path_folder, path, 'v_var', f'root_{col_name}', f'derivative_root_{col_name}')
                    self.calculate_derivative(path_folder, path, 'v_var', f'log10_{col_name}', f'derivative_log_{col_name}')
                    self.calculate_mobility(path_folder, path, col_name)

                    # self.calculate_saturation_max_current_values(path_folder , path, ['v_var' , 'ids', f'log10_{col_name}' ,'root_ids' , 'derivative_root_ids'])

                    for sweep in ['back', 'forth']:
                        self.define_region_aux(path_folder, path, sweep)
                        self.define_region(path_folder, path, sweep)
                        # self.calculate_max_current(path_folder , path ,sweep)


    def draw_saturation_curves(self , path_folder = 'Vg', axes=None):
        # print(self.dots_diff_regions_df)
        for path in self.tables[path_folder]:
            for sweep in ['back', 'forth']:
                self.saturacion_curve(path_folder, path, 'root_ids' , 'derivative_log_ids', sweep= sweep, folder=self.folder, axes=axes)


    def create_graphic(self, path_folder, y_col, sweep, graphic_name, invert_x=1, invert_y=1, log_scale=False, derivative = False):
        
        #label voltaje
        label_voltaje = {'Vds' : 'Vg'  , 'Vg':'Vds'}
        sns.set_theme(style="darkgrid")

        
        # Create a figure and axis
        figure, ax = plt.subplots(figsize=(8, 6))

        for path in self.tables[path_folder]:
            table_test = self.tables[path_folder][path]

            if derivative == True:
                derivative_filter = invert_x*table_test['v_var'] < max(abs(table_test['v_var'])) 
                table_test = table_test[derivative_filter]

            if sweep is not None:
                table_test = table_test[table_test['sweep'] == sweep]

            # Apply log scale to y-axis if needed
            if log_scale:
                ax.set_yscale("log")

            # Plot the line with or without confidence interval
            sns.lineplot(x=invert_x * table_test['v_var'], y=invert_y * table_test[y_col], label=f" {label_voltaje[path_folder]}= {path[-21:-17]} (V)")

        # Add labels and title

        plt.xlabel(f"{'-' if invert_x == -1 else ''}{path_folder} (V)")
        plt.ylabel(f"{y_col} (log scale)" if log_scale else f"{y_col} (unit)")
        plt.title('Graph ' + graphic_name)

        # Add legend if needed
        plt.legend()

        # Save the figure
        self.graphics[graphic_name] = figure
        plt.savefig(self.graphic_folder + graphic_name + ('_log_scale' if log_scale else '') + '.png')
        plt.close(figure)


    def draw_graphics(self, folder = 'graphics'):
        self.graphic_folder = folder + 'caracteristic_curves/'
        os.makedirs(self.graphic_folder, exist_ok = True)

        self.create_graphic('Vds' , 'ids' , 'back', 'outer_back', invert_x=-1 , invert_y=-1)
        self.create_graphic('Vds' , 'ids' , 'forth', 'outer_forth', invert_x=-1 , invert_y=-1)
        
        self.create_graphic('Vg' , 'ids' , 'back', 'transference_back' , invert_x=-1 , invert_y=-1)
        self.create_graphic('Vg' , 'ids' , 'forth', 'transference_forth', invert_x=-1 , invert_y=-1)

        self.create_graphic('Vds' , 'log10_ids' , 'back', 'outer_log_back', invert_x=-1 )
        self.create_graphic('Vds' , 'log10_ids' , 'forth', 'outer_log_forth', invert_x=-1 )
        
        self.create_graphic('Vg' , 'log10_ids' , 'back', 'transference_log_back', invert_x=-1 )
        self.create_graphic('Vg' , 'log10_ids' , 'forth', 'transference_log_forth', invert_x=-1 )

        self.create_graphic('Vg' , 'derivative_ids' , 'back', 'Rg_back', invert_x=-1 , log_scale=False , derivative=True)
        self.create_graphic('Vg' , 'derivative_ids' , 'forth', 'Rg_forth', invert_x=-1 , log_scale=False , derivative=True)
        self.create_graphic('Vg' , 'derivative_ids' , 'back', 'Rg_log_back', invert_x=-1 , log_scale=True , derivative=True)
        self.create_graphic('Vg' , 'derivative_ids' , 'forth', 'Rg_log_forth', invert_x=-1 , log_scale=True , derivative=True)

        self.create_graphic('Vds' , 'd_ids/d_Vds' , 'back', 'Rch_back', invert_x=-1 , log_scale=False , derivative=True)
        self.create_graphic('Vds' , 'd_ids/d_Vds' , 'forth', 'Rch_forth', invert_x=-1 , log_scale=False , derivative=True)
        self.create_graphic('Vds' , 'd_ids/d_Vds' , 'back', 'Rch_log_back', invert_x=-1 , log_scale=True , derivative=True)
        self.create_graphic('Vds' , 'd_ids/d_Vds' , 'forth', 'Rch_log_forth', invert_x=-1 , log_scale=True , derivative=True)

        self.graphic_folder = folder + 'mobility/'
        os.makedirs(self.graphic_folder, exist_ok = True)

        self.create_graphic('Vg' , 'mobility_sat' , 'back', 'mobility_back', invert_x=-1 , derivative=True)
        self.create_graphic('Vg' , 'mobility_sat' , 'forth', 'mobility_forth', invert_x=-1, derivative=True )


    def create_3d_graphic(self, path_folder = 'Vg'):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        for path in self.tables[path_folder]:
            df = self.tables[path_folder][path]

            ax.plot(-df['v_var'], -df['v_fixed'], -df['ids'], c='blue')
        plt.savefig(self.graphic_folder + str(round(self.L,0)) + '.png')


    def determinate_mobility(self , path_folder, path, sweep):
        condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
        condition_region = self.tables[path_folder][path]['region'] == 'saturacion'
        x = self.tables[path_folder][path][condition_sweep & condition_region]['v_var'].values
        y = self.tables[path_folder][path][condition_sweep & condition_region]['root_ids'].values

        linear_regresion= determinate_linear_regresion_2(x,y)

        #define the variables 
        m =  linear_regresion['slope']
        b = linear_regresion['intercept']
        se_m =linear_regresion['se_slope']
        se_b = linear_regresion['se_intercept']
        r2 = linear_regresion['r_value']

        #calculate the voltaje vth
        valores_vth = [m, b]
        incertidumbres_vth = [se_m, se_b]

        def operacion_vth(m,b): 
            return -b/m
            
        vth , error_vth = calcular_incertidumbre_operacion(operacion_vth, valores_vth, incertidumbres_vth)


        # calculate mu saturacion
        valores_mu = [self.mobility_factor, m]
        incertidumbres_mu = [self.incertidumbre_mobility_factor, se_m]

        def operacion_mu(mobility_factor, m):
            return (2*mobility_factor)*(m**2)
        
        mu , error_mu  = calcular_incertidumbre_operacion(operacion_mu, valores_mu, incertidumbres_mu)

        return [m, b,r2, vth, error_vth, mu, error_mu]


    def determinate_subumbral_voltaje(self , path_folder, path, sweep):
        condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
        condition_region = self.tables[path_folder][path]['region'] == 'saturacion'
        x = self.tables[path_folder][path][condition_sweep & condition_region]['derivative_log_ids'].values

        def operacion_s (ids_):
            return -1./ids_

        valores_s = [min(x)]
        incertidumbre_s = [np.log10(abs(min(x)))*0.5]

        S , error_S  =  calcular_incertidumbre_operacion(operacion_s, valores_s , incertidumbre_s)

        kb = 1.380649 * 1e-23
        q = 1.6*1e-19
        const = kb*q*np.log(10)

        def op_aux_Nit(s, T ,C):
            N_it =  s * C /T
            return  N_it 

        N_aux_it , error_N_aux_it = calcular_incertidumbre_operacion(op_aux_Nit, [S , self.T , self.C] , [error_S , self.se_T ,self.se_C ]) / const

        def op_Nit(N_aux_it ,C ):
            return N_aux_it - (C / (q**2))

        N_it , error_N_it = calcular_incertidumbre_operacion(op_Nit, [N_aux_it , self.C] , [error_N_aux_it ,self.se_C ])
        return [S , error_S , N_it , error_N_it]


    def determinate_gain_ratio(self, path_folder, path, sweep):
        try:
            condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
            table = self.tables[path_folder][path][condition_sweep]

            condition_max_root = table['root_ids'] == max(table['root_ids'])
            condition_min_log = table['derivative_log_ids'] == min(table['derivative_log_ids'])

            ids_max = table.loc[condition_max_root, 'ids'].values
            ids_min = table.loc[condition_min_log, 'ids'].values

            if len(ids_max) == 0 or len(ids_min) == 0:
                # Manejar el caso de listas vacías (puedes personalizar el manejo del error aquí)
                raise ValueError(f"ids_max o ids_min it's empty on the path : {path} - {sweep}")

            gain = [ids_max, ids_min]
            return gain

        except Exception as e:
            # Manejar cualquier otra excepción que pueda ocurrir
            print(f"Error en determinate_gain_ratio: {e}")
            return [] ,[]


    def determinate_parameters(self, path_folder = 'Vg'):
        parameters = []
        for sweep in ['back', 'forth']:
            for path in self.tables[path_folder]:
                condition_sweep = self.tables[path_folder][path]['sweep'] == sweep
                vds = round(self.tables[path_folder][path][condition_sweep]['v_fixed'].values.mean(),1)
                charge_carriers = self.determinate_mobility( path_folder, path, sweep)
                trap_states = self.determinate_subumbral_voltaje(path_folder, path, sweep)
                ids_max , ids_min = self.determinate_gain_ratio(path_folder, path, sweep)


                ids_max_value = ids_max[0] if ids_max else None
                ids_min_value = ids_min[0] if ids_min else None
                ratio_value = ids_max_value / ids_min_value if ids_max_value is not None and ids_min_value is not None else None

                list_parameters = [vds, sweep] + charge_carriers + trap_states + [ids_max_value, ids_min_value, ratio_value] + [self.W , self.L , self.C, self.T] 
                parameters.append(list_parameters)


        list_column = ['Vds', 'sweep', 'm', 'b', 'r2', 'vth','se_vth' ,'mu_sat' , 'se_mu_sat','S','se_S', 'N_it', 'se_N_it',  '$I_{on}$' , '$I_{off}$', '$I_{on}/I_{off}$','W', 'L', 'C', 'T']
        df_parameters = pd.DataFrame(parameters, columns = list_column)
        df_parameters['E'] = df_parameters['Vds'] / df_parameters['L']
        self.parameters = df_parameters


    def draw_parameters(self, x_parameter='Vds', sweep='back', folder='graphics/', exlusion_vds_values = [0.0]):
        condition_sweep = self.parameters['sweep'] == sweep
        condition_exclusion = ~self.parameters['Vds'].isin(exlusion_vds_values)
        table = self.parameters[condition_sweep & condition_exclusion]

        for parameter in ['vth', 'mu_sat', 'S']:
            if parameter == '$I_{on}/I_{off}$' :
                parameter_label = 'gain_ratio'
            else:
                parameter_label = parameter

            fig, ax = plt.subplots()
            ax.set_ylabel(parameter_label)
            ax.set_xlabel(x_parameter)

            ax.errorbar(table[x_parameter], table[parameter], yerr=table[f"se_{parameter}"] )

            graphic_name = f"{sweep}_{parameter}_{x_parameter}"
            self.graphics[graphic_name] = fig
            self.graphic_folder = os.path.join(folder, 'parameters')
            os.makedirs(self.graphic_folder, exist_ok=True)
            fig.savefig(os.path.join(self.graphic_folder, f"{graphic_name}.png"))
            plt.close(fig)


    def determinate_electric_model_parameters(self , sweep = 'back', n_exp_electric = 0.5):

        condition_sweep = self.parameters['sweep'] == sweep
        x = self.parameters[condition_sweep]['Vds']
        root_E = abs(x / self.L)**n_exp_electric

        vth = self.parameters[condition_sweep]['vth']
        mu_sat = self.parameters[condition_sweep]['mu_sat']
        log_mu_sat = np.log(abs(mu_sat))

        vth_E =  determine_quadratic_regression(root_E , vth)
        roots_vth_E = calculate_roots(vth_E[:-1])
        vertice_vth_E = calculate_vertice(vth_E[:-1])

        log_mu_sat_E =  determine_quadratic_regression(root_E , log_mu_sat)
        roots_log_mu_sat_E = calculate_roots(log_mu_sat_E[:-1])
        vertice_log_mu_sat_E = calculate_vertice(log_mu_sat_E[:-1])

        #slope por calcular los vertices [expresarlos como vth, log_mu, mu] y raices, y expresarlos como root_E , E , vds

        electric_model_parameters = vth_E + roots_vth_E + vertice_vth_E + log_mu_sat_E + roots_log_mu_sat_E + vertice_log_mu_sat_E

        self.electric_model.append(electric_model_parameters)
        self.electric_model = pd.DataFrame(self.electric_model)
        self.electric_model['L'] = self.L
        self.electric_model.columns = ['a_vth', 'b_vth', 'c_vth', 'r2_vth' ,'root_one_vth' ,'root_two_vth','vth_critic', 'sqrt_E_critic', 'a_log_mu_sat', 'b_log_mu_sat', 'c_log_mu_sat', 'r2_log_mu_sat' , 'roots_one_log_mu_sat', 'roots_two_log_mu_sat','log_mu_sat_critic' , 's' , 'L']


    def determinate_mean_parameters(self, sweep = 'back'):
        condition_sweep = self.parameters['sweep'] == sweep
        S =  self.parameters[condition_sweep]['S']
        N_it =  self.parameters[condition_sweep]['N_it']

        mean_parameters = [S.mean() ,N_it.mean() ]
        sigma_parameters = [S.std() ,N_it.std() ]
        return mean_parameters


    def export_tables(self, path_folder = 'Vg', excel_name = 'excel_file_with_the_data.xlsx'):
        with pd.ExcelWriter(excel_name) as writer:
            counter = 0
            for path in self.tables[path_folder]:
                df = self.tables[path_folder][path]
                df.to_excel(writer, sheet_name = str(counter) , index=False)
                counter += 1


    def analizy_ofet(self, parameter_ , folder_data_path ,relative_path , set_folder, export_parameters = False , only_graphics = False):
    
        self.process_txt_files(folder_data_path + relative_path)
        self.folder = f"{set_folder}/graphics/{relative_path}"
        self.create_caracteristics()
        self.draw_saturation_curves(path_folder = 'Vg', axes=None)
        self.draw_graphics(self.folder)

        if only_graphics == False:

            self.create_3d_graphic()

            self.determinate_parameters()
            self.draw_parameters(folder=f"{set_folder}/graphics/{relative_path}", sweep = 'back')
            self.draw_parameters(folder=f"{set_folder}/graphics/{relative_path}", sweep = 'forth')
            self.export_tables(excel_name = f"{set_folder}/data_{parameter_}.xlsx")
            
            self.determinate_electric_model_parameters()
            self.determinate_mean_parameters()

            os.makedirs(set_folder, exist_ok = True)

            if export_parameters == True:
                self.parameters.to_excel(f'{set_folder}/parameters_{parameter_}.xlsx' , index= False)
                self.electric_model.to_excel(f'{set_folder}/electric_parameters_{parameter_}.xlsx')


class ofet_set:

    def __init__(self, set_ofets ,name_folder = 'group') -> None:
        self.ofets = set_ofets
        self.name_folder = name_folder

        if not isinstance(set_ofets, list):
            print("'set_ofets' needs to be a list")
            
        self.ofets_parameters = None
        self.graphic_folder = name_folder +'/graphics/'
        self.thermical_model = []


    def add_ofet(self, ofet_element):   
        if isinstance(ofet_element, list):
            self.ofets.extend(ofet_element)
        else:
            pass


    def organize_table_parameters(self, parameter_of_study = None ):
        list_of_parameters =  []
        for i , each_ofet in enumerate(self.ofets):
            if parameter_of_study is not None : 
                key = next(iter(parameter_of_study))  # Obtén la clave ('T' en este caso)
                value = parameter_of_study[key][i]   # Obtén el valor correspondiente de T_list
                each_ofet.parameters.loc[:, key] = value

            list_of_parameters.append(each_ofet.parameters)

        self.ofets_parameters = pd.concat(list_of_parameters , axis = 0 ).reset_index(drop=True)

        self.ofets_parameters.to_excel(f"{self.name_folder}/parameters.xlsx" , index = False )


    def transform_table_parameters(self):
        self.ofets_parameters['sqrt_T^{-1}'] = np.sqrt(1/self.ofets_parameters['T'])
        self.ofets_parameters['sqrt_{E}'] = np.sqrt(abs(self.ofets_parameters['E']))
        self.ofets_parameters.to_excel(f"{self.name_folder}/parameters.xlsx" , index = False )


    def organize_electric_model(self, folder = '/', parameter = None):
        list_electric_parameters = []
        for i, each_ofet in enumerate(self.ofets):
            if parameter is not None : 
                key = next(iter(parameter))  # Obtén la clave ('T' en este caso)
                value = parameter[key][i]   # Obtén el valor correspondiente de T_list
                each_ofet.electric_model.loc[:, key] = value

                
            list_electric_parameters.append(each_ofet.electric_model)
        
        self.electric_model = pd.concat(list_electric_parameters, axis=0).reset_index(drop=True)
        self.electric_model.to_excel( folder + '/electric_parameters.xlsx', index=False)



    def calculate_E0_table(self, folder = '/'):
        E0_L_list = pd.DataFrame()
        L_columns = []

        for L in self.ofets_parameters['L'].unique():
            condition_L = self.ofets_parameters['L'] == L
            table = self.ofets_parameters[condition_L]
            E = table['E']
            mu = table['mu_sat']
            sweep = table['sweep']
            vds = table['Vds']

            E0_list = []
            L_columns.append(L)
            for E_i, mu_i, E_i1, mu_i1, vds_ , sweep_ in zip(E, mu, E[1:], mu[1:], vds ,sweep):
                E_0 = (E_i * E_i1) * ((mu_i1 - mu_i) / (E_i1 * mu_i1 - E_i * mu_i))
                E0_list.append([E_0, vds_, sweep_ , L])
    
    
 
            E0_L_list = pd.concat( [E0_L_list , pd.DataFrame(E0_list)] ,axis=0)
        self.E0_L_list  =  pd.DataFrame(E0_L_list)
        self.E0_L_list.columns =  ['E0' , 'Vds' , 'sweep' , 'L']
        self.E0_L_list.to_excel(f"{folder}/E0_L.xlsx")




    def calculate_E0(self):
        df = self.E0_L_list.loc[ :-1]
        E0_mean = df.mean()
        E0_std = df.std()
        self.E0 = [E0_mean , E0_std]
        print(self.E0)




    def create_recursive_graphic(self, y_parameter, x_parameter='Vds', sweep='back', 
                                 id_parameter='L' , exclusion = {'parameter' : 'Vds' , 'values' : [0.0]},
                                 n_legend_col = 5 , log_scale = False):
        unit = {'Vds' : '(V)' , 'L' : '($\mu$m)' , 'T' : '(K)' ,'sqrt_T^{-1}' : '(mK)^{-1}' }
        scale_ = {'Vds' : 1 , 'L': 1e6 , 'T': 1 , 'sqrt_T^{-1}' : 1e3}

        sns.set(style="white")  # Configuración opcional de estilo Seaborn
        figure, ax = plt.subplots(figsize=(10,8))

        for each_parameter in set(self.ofets_parameters[id_parameter]):
            condition_exclusion = ~self.ofets_parameters[exclusion['parameter']].isin(exclusion['values'])
            condition_id_parameter = self.ofets_parameters[id_parameter] == each_parameter
            condition_sweep = self.ofets_parameters['sweep'] == sweep
            table = self.ofets_parameters[condition_id_parameter & condition_sweep & condition_exclusion]

            # Utilizando Matplotlib para las barras de error
            ax.errorbar(x=table[x_parameter], y=table[y_parameter], yerr=table[f"se_{y_parameter}"], label=f"{id_parameter}={round(scale_[id_parameter]*each_parameter,1)} {unit[id_parameter]}")

        ax.set_xlabel(x_parameter)
        ax.set_ylabel(y_parameter)
        
        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol= n_legend_col)

        plt.title(f"{y_parameter} {sweep} {x_parameter}")
        
        log_label = ''
        if log_scale:
            ax.set_yscale('log')
            log_label = 'log'

        if x_parameter in ['Vds', 'E' ,'T', 'sqrt_T^{-1}']:
            plt.gca().invert_xaxis()

        plt.savefig(self.graphic_folder + log_label + f"{y_parameter}_{sweep}_{x_parameter}")
        plt.close(figure)



    def draw_recursive_graphics(self, id_parameter='L', parameters_to_graphic = ['vth', 'mu_sat', 'S'], log_scale = False):
        os.makedirs(self.graphic_folder, exist_ok = True)
        n_col = 5
        for i_parameter in parameters_to_graphic:
            if i_parameter == 'S':
                n_col = 4

            self.create_recursive_graphic(i_parameter, sweep='back', id_parameter=id_parameter, n_legend_col= n_col , log_scale=log_scale)
            self.create_recursive_graphic(i_parameter, sweep='forth', id_parameter=id_parameter, n_legend_col=n_col, log_scale=log_scale)




    def draw_inverse_graphics(self, x_parameter = 'L' , parameters_to_graphic = ['vth', 'mu_sat', 'S']):
        os.makedirs(self.graphic_folder, exist_ok = True)
        for i_parameter in parameters_to_graphic:
            self.create_recursive_graphic(i_parameter, sweep='back' , x_parameter = x_parameter ,id_parameter = 'Vds')
            self.create_recursive_graphic(i_parameter, sweep='forth' , x_parameter = x_parameter ,id_parameter = 'Vds')





    def draw_histeresis_curve(self, vds , path_folder = 'Vg', y_parameter = 'ids' , invert_x = -1 , invert_y =-1 ):
        figure , ax = plt.subplots()

        for each_ofet in self.ofets :
            for path in each_ofet.tables[path_folder]:
                vds_ = round(each_ofet.tables[path_folder][path]['v_fixed'].values.mean(),1)
                if vds_== vds :
                    x = invert_x*each_ofet.tables[path_folder][path]['v_var']
                    y = invert_y*each_ofet.tables[path_folder][path][y_parameter]
                    ax.plot(x,y, label = f" vds = {vds} (V)")
                

        ax.legend(loc='upper left', bbox_to_anchor=(0, 1.1), ncol= 5)

        plt.title(f"Histeresis vds = {vds} (V)")

        ax.set_xlabel(f"{path_folder} (V)")
        ax.set_ylabel(f"{y_parameter} (A)")

        
        plt.savefig(self.graphic_folder + 'histeresis_{}.png'.format(vds))
        plt.close(figure)






    def draw_3d_graphics(self , path_folder = 'Vg' , folder ='' ):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        color = ['g', 'r' , 'b']
        for i, ofet in enumerate(self.ofets) :
            for path in ofet.tables[path_folder]:
                df = ofet.tables[path_folder][path]

                ax.plot(-df['v_var'], -df['v_fixed'], -df['ids'], c = color[i])
        plt.savefig(f"{folder}/graphics/3d_curves.png")
        plt.close()



    def calculate_thermical_model(self, sweep = ['forth', 'back']):
        for sweep_ in sweep:
            condition_sweep = self.ofets_parameters['sweep'] == sweep_
            sqrt_T = self.ofets_parameters[condition_sweep]['sqrt_T^{-1}']
            log_mu_sat = np.log(abs(self.ofets_parameters[condition_sweep]['mu_sat']))

            log_mu_sat_T =  determine_quadratic_regression(sqrt_T, log_mu_sat)
            roots_log_mu_sat_T = calculate_roots(log_mu_sat_T[:-1])
            vertice_log_mu_sat_T = calculate_vertice(log_mu_sat_T[:-1])


            thermical_model_parameters = log_mu_sat_T + roots_log_mu_sat_T + vertice_log_mu_sat_T + [sweep_]
            self.thermical_model.append(thermical_model_parameters)
        self.thermical_model = pd.DataFrame(self.thermical_model)
        self.thermical_model.columns = ['a_mu_sqrt_T', 'b_mu_sqrt_T', 'c_mu_sqrt_T', 'r2_mu_sqrt_T' ,'root_one_mu_sqrt_T' ,'root_two_mu_sqrt_T','h_mu_sqrt_T', 'k_mu_sqrt_T' ,'sweep']

        self.thermical_model.to_excel(f"{self.name_folder}/thermal_parameters.xlsx" , index = False )


    def calculate_mixed_model(self, variable_name, x_name, y_name , folder = ''):
        self.ofets_parameters['log_mu'] = np.log(abs(self.ofets_parameters['mu_sat']))

        equation, coefficients = solve_system(self.ofets_parameters , variable_name, x_name, y_name)

        self.mixted_model = [equation, pd.DataFrame(coefficients)]
        self.mixted_model[1].T.to_excel(f"{folder}/mixed_model.xlsx" , index=False)

        print(equation)



    #now i want to calculate the values of mu using the model and later the value of Ids... 



class CompactGroupTOfets:
    def __init__(self, T_ofets, set_folder, T_list):
        self.group_T_ofets = ofet_set(T_ofets, name_folder=set_folder)
        self.T_list = T_list
        self.set_folder = set_folder

    def run_analysis(self):
        self._organize_table_parameters()
        self._transform_table_parameters()
        self._organize_electric_model()
        self._draw_graphics()
        self._draw_inverse_graphics()
        self.group_T_ofets.draw_histeresis_curve(-5.0)
        # self.group_T_ofets.draw_3d_graphics(folder=self.set_folder)
        self._calculate_E0_table()
        # self.group_T_ofets.calculate_E0()
        self.group_T_ofets.calculate_thermical_model()
        self.group_T_ofets.calculate_mixed_model('log_mu', 'sqrt_{E}', 'sqrt_T^{-1}', folder=self.set_folder)

    def _organize_table_parameters(self):
        self.group_T_ofets.organize_table_parameters(parameter_of_study={'T': self.T_list})

    def _transform_table_parameters(self):
        self.group_T_ofets.transform_table_parameters()

    def _organize_electric_model(self):
        self.group_T_ofets.organize_electric_model(folder=self.set_folder, parameter={'T': self.T_list})

    def _draw_graphics(self):
        self.group_T_ofets.draw_recursive_graphics(id_parameter='T', parameters_to_graphic=['vth', 'mu_sat', 'S'])
        self.group_T_ofets.draw_recursive_graphics(id_parameter='sqrt_T^{-1}', parameters_to_graphic=['vth', 'mu_sat', 'S'])
        self.group_T_ofets.draw_recursive_graphics(id_parameter='sqrt_T^{-1}', parameters_to_graphic=['mu_sat'], log_scale=True)

    def _draw_inverse_graphics(self):
        self.group_T_ofets.draw_inverse_graphics(x_parameter='T')
        self.group_T_ofets.draw_inverse_graphics(x_parameter='sqrt_T^{-1}')

    def _calculate_E0_table(self):
        self.group_T_ofets.calculate_E0_table(folder=self.set_folder)





