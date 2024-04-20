#!/usr/bin/env python
# coding: utf-8

# ### A Machine Learning Calibration Model For Correcting Low-Cost PM2.5 Sensor Data For Air Quality Monitoring. - RF Implementation

# In[1]:


# load relevant libraries

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import math
from math import sqrt
from scipy.interpolate import interpn
import datetime
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from matplotlib import colors as mpc
from matplotlib.colors import Normalize
from matplotlib import cm
import pprint
import seaborn as sns; sns.set()
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV

from IPython.display import clear_output

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[100]:


# Import air quality dataset
data_h = pd.read_csv("DailyPA_Ref_Mon.csv", header=0, index_col=0)

# Subset of dataset to get columns for model
df_ = data_h[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
df_.head()


# In[101]:


# Define functions for model evaluation
def n_bias(estimated, true):
    estimated_ = np.array(estimated).reshape(-1)
    true_ = np.array(true).reshape(-1)
    return np.sum(np.subtract(estimated_, true_)) / estimated_.shape[0]

def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))


# In[102]:


# Summary statistics of raw data
print('R2: %.3f' % r2_score(df_['purpleair_pm2.5'], df_['RefEmb_pm2.5']))
print('MAE:  %.3f' % mean_absolute_error(df_[['purpleair_pm2.5']], df_[["RefEmb_pm2.5"]]))
print('RMSE: %.3f' % sqrt(mean_squared_error(df_['purpleair_pm2.5'], df_['RefEmb_pm2.5'])))
print('Bias: %.3f' % np.mean(df_['purpleair_pm2.5'] - df_['RefEmb_pm2.5'])) # Bias (mean difference)


# ### Model Implementation

# In[201]:


# Splitting the data into features (X) and target (y)
x = df_[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
y = df_["RefEmb_pm2.5"]

# Train-test dataset preparation
dataset_size = len(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Grid search regressor parameters configuration
start_time = datetime.datetime.now()
rf_reg = RandomForestRegressor(random_state = 0, warm_start = True)

# Perform a grid search to find the best hyperparameters
parameters = {
                "n_estimators"      : [100,200],
                "max_features"      : ["auto","sqrt","log2"],
                "min_samples_split" : [1,2,4,5],
                "bootstrap": [True, False],
}

grid_search = GridSearchCV(estimator=rf_reg, param_grid=parameters, cv=10, n_jobs=-1, verbose=1, scoring='r2')

# Printing the results of grid search
print("parameters:")
pprint.pprint(parameters)
grid_search.fit(x_train, y_train)
print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
end_time = datetime.datetime.now()
print ('Computing Time: %d' % ((end_time - start_time).seconds))

# Make predictions on the testing set with the best hyperparameters
y_predrf = grid_search.predict(x_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_predrf)
r2 = r2_score(y_test, y_predrf)
rmse_value = rmse(y_test, y_predrf)
bias = n_bias(y_test, y_predrf)

print('PM2.5c = X1*PAPM2.5 + X2*TEMP + X3*RH')
print(' R2 = %.3f ' % r2)
print(' MAE = %.3f ' % mae)
print(' RMSE = %.3f ' % rmse_value)
print('bias = %.3f ' % bias)


# In[151]:


# density_scatter plot function 
def density_scatter( x , y, ax = None, sort = True, bins = 20, figsize=(6,4), **kwargs )   :
   
    """Scatter plot colored by 2d histogram """

    if ax is None :
        fig , ax = plt.subplots(figsize=figsize)
        
    # Compute 2D histogram    
    data , x_e, y_e = np.histogram2d( x, y, bins = bins)
    z = interpn( ( 0.5*(x_e[1:] + x_e[:-1]) , 0.5*(y_e[1:]+y_e[:-1]) ) , data , np.vstack([x,y]).T , method = "splinef2d", bounds_error = False )

    z[np.where(np.isnan(z))] = 0.0      # Handle NaN values by setting them to 0.0
    
    # Sort the points by density and setup colormap and normalization for coloring the scatter plot
    if sort :
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
    cmap = matplotlib.cm.get_cmap('jet')
    normalize = matplotlib.colors.Normalize(vmin=0, vmax=max(z))
    
    # Scatter plot with colors based on density
    ax.scatter( x, y, c=z, **kwargs, cmap=cmap)
    
    # Create a color bar for density
    cax, _ = matplotlib.colorbar.make_axes(ax)
    #norm = Normalize(vmin = np.min(z), vmax = np.max(z))
    cbar = matplotlib.colorbar.ColorbarBase(cax, cmap=cmap, norm=normalize)
    #cbar = fig.colorbar(cm.ScalarMappable(norm=norm), ax=ax)
    cbar.ax.set_ylabel('Density')
    
    # Fit a linear regression line (y = mx + c)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    print(z)
    
    # Plot the regression line
    ax.plot(x,p(x),"r-")
    
    # Plot the y=x line for reference
    ax.plot(x,x,":",color='grey')
    
    return ax,fig,z


# In[205]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predrf, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
pol_text = 'y = %.2fx + %.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('RF Regressor Modeling', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight = "bold")
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight = "bold")
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('grey')
    spine.set_linewidth(0.25)

ax.patch.set_facecolor('white')
ax.set_ylim(0, 80)
ax.set_xlim(0, 80)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('random_forest_regressor_plot.png', dpi=300)


# ### ANALYSIS OF MISSING INPUTS

# #### 1- Missing Relative Humidity (RH) values

# In[188]:


# Condition rf over PurpleAir PM2.5 and temperature

grid_search.fit(x_train[["purpleair_pm2.5", "avg_temp"]], x_train[["RefEmb_pm2.5"]])
y_predrf = grid_search.predict(x_test[["purpleair_pm2.5", "avg_temp"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predrf)
r2 = r2_score(y_test, y_predrf)
rmse_value = sqrt(mean_squared_error(y_test, y_predrf))
bias = n_bias(y_test, y_predrf)

print('PM2.5c = X1*PAPM2.5 + X2*Temp')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[189]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predrf, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
pol_text = 'y = %.2f x+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('RF Regressor Without RH', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight = "bold")
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight = "bold")
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('grey')
    spine.set_linewidth(0.25)

ax.patch.set_facecolor('white')
ax.set_ylim(0, 120)
ax.set_xlim(0, 120)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('random_forest_regressor-RH_plot.png', dpi=300)


# #### 2- Missing Temperature values

# In[195]:


# Condition rf over PurpleAir PM2.5 and RH

grid_search.fit(x_train[["purpleair_pm2.5", "RH"]], x_train[["RefEmb_pm2.5"]])
y_predrf = grid_search.predict(x_test[["purpleair_pm2.5", "RH"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predrf)
r2 = r2_score(y_test, y_predrf)
rmse_value = sqrt(mean_squared_error(y_test, y_predrf))
bias = n_bias(y_test, y_predrf)

print('PM2.5c = X1*PAPM2.5 + X2*RH')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[199]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predrf, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
pol_text = 'y = %.2f x+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('RF Regressor Without Temperature', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight = "bold")
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight = "bold")
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('grey')
    spine.set_linewidth(0.25)

ax.patch.set_facecolor('white')
ax.set_ylim(0, 70)
ax.set_xlim(0, 70)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('random_forest_regressor-Temp_plot.png', dpi=300)


# #### 3- Missing Temperature and Relative Humidity (RH) values

# In[206]:


# Condition rf over PurpleAir PM2.5

grid_search.fit(x_train[["purpleair_pm2.5"]], x_train[["RefEmb_pm2.5"]])
y_predrf = grid_search.predict(x_test[["purpleair_pm2.5"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predrf)
r2 = r2_score(y_test, y_predrf)
rmse_value = sqrt(mean_squared_error(y_test, y_predrf))
bias = n_bias(y_test, y_predrf)

print('PM2.5c = X1*PAPM2.5')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[207]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predrf, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
pol_text = 'y = %.2f x+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('RF Regressor Without Temp & RH', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight = "bold")
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight = "bold")
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)

# Set the border color and width
for spine in ax.spines.values():
    spine.set_edgecolor('grey')
    spine.set_linewidth(0.25)

ax.patch.set_facecolor('white')
ax.set_ylim(0, 70)
ax.set_xlim(0, 70)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('random_forest_regressor-Temp-RH_plot.png', dpi=300)


# In[23]:


import sys
print(sys.version)


# In[ ]:




