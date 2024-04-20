#!/usr/bin/env python
# coding: utf-8

# ### A Machine Learning Calibration Model For Correcting Low-Cost PM2.5 Sensor Data For Air Quality Monitoring. - XGBoost Implementation

# In[3]:


# Loading the necessary libraries
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from scipy.interpolate import interpn
import datetime
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as mpc
from matplotlib.colors import Normalize
from matplotlib import cm
import math
from math import sqrt
import pprint
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBRegressor
import xgboost as xgb

from IPython.display import clear_output

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# Import air quality dataset
data_h = pd.read_csv("DailyPA_Ref_Mon.csv", header=0, index_col=0)

# Subset of dataset to get columns for model
df_ = data_h[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
df_.head()


# In[6]:


df = data_h[["purpleair_pm2.5", "RefEmb_pm2.5"]]
df.describe()


# In[41]:


# Define functions for model evaluation
def n_bias(estimated, true):
    estimated_ = np.array(estimated).reshape(-1)
    true_ = np.array(true).reshape(-1)
    return np.sum(np.subtract(estimated_, true_)) / estimated_.shape[0]

def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted))


# In[42]:


### Summary statistics of raw data
print('R-squared: %.3f' % r2_score(df_['purpleair_pm2.5'], df_['RefEmb_pm2.5']))
print('MAE:  %.3f' % mean_absolute_error(df_[['purpleair_pm2.5']], df_[["RefEmb_pm2.5"]]))
print('RMSE: %.3f' % sqrt(mean_squared_error(df_['purpleair_pm2.5'], df_['RefEmb_pm2.5'])))
print('Bias: %.3f' % np.mean(df_['purpleair_pm2.5'] - df_['RefEmb_pm2.5'])) # Bias (mean difference)


# In[52]:


df_ = data_h[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
selected_features = ['purpleair_pm2.5','RefEmb_pm2.5']
df_selected = df_[selected_features]

# Calculate Spearman correlation matrix
correlation_matrix = df_selected.corr(method='pearson')

# Generate a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Spearman's Correlation Coefficient Heatmap")
plt.show()


# ### XGBoost Model Implementation

# In[43]:


# Splitting the data into features (X) and target (y)
x = df_[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
y = df_["RefEmb_pm2.5"]

# Train-test dataset preparation
dataset_size = len(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Grid search regressor parameters configuration
start_time = datetime.datetime.now()
xgb_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Perform a grid search to find the best hyperparameters
parameters = {
    'n_estimators': [200, 300],
    'max_depth': [5, 7],
    'learning_rate': [0.01, 0.05],
}

grid_search = GridSearchCV(estimator=xgb_reg, param_grid=parameters, cv=10, n_jobs=-1, verbose=1, scoring='r2')

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
y_predxgb = grid_search.predict(x_test)

# Evaluating the model
mae = mean_absolute_error(y_test, y_predxgb)
r2 = r2_score(y_test, y_predxgb)
rmse_value = sqrt(mean_squared_error(y_test, y_predxgb))
bias = n_bias(y_test, y_predxgb)

print('PM2.5c = X1*PAPM2.5 + X2*TEMP + X3*RH')
print(' R2 = %.3f ' % r2)
print(' MAE = %.3f ' % mae)
print(' RMSE = %.3f ' % rmse_value)
print(' Bias = %.3f ' % bias)


# In[44]:


# Plotting time series for raw data and reference data
plt.figure(figsize=(12,6))
plt.plot(df_.index, df_['purpleair_pm2.5'], label='PurpleAir PM2.5')
plt.plot(df_.index, df_['RefEmb_pm2.5'], label='Reference PM2.5', color = 'r')
plt.xlabel("Date", fontsize = 15, fontweight = 'bold')
plt.ylabel("PM2.5 (ug /m³)", fontsize = 15, fontweight = 'bold')
plt.legend(loc='upper left')

plt.savefig('Raw-Ref_Data_plot.png', dpi=300)

plt.show()


# In[ ]:





# In[45]:


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


# In[46]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predxgb, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
legend_fontsize = 'medium'
pol_text = 'y = %.2fx+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('XGBoost Regressor Modeling', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight='bold')
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight='bold')
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.patch.set_facecolor('white')
ax.set_ylim(0, 80)
ax.set_xlim(0, 80)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('xgboost_regressor_plot.png', dpi=300)


# ### Correcting new sensor measurements

# In[45]:


# Import your raw dataset
new_sensor_reading = pd.read_csv("RawData.csv", header=0, index_col=0)

# Ensure that your raw dataset has the same features as the original dataset
new_sensor_reading = new_sensor_reading[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]


# In[55]:


# Calibrating new sensor readings


# Use the trained model to make predictions on the raw data
corrected_data = grid_search.predict(new_sensor_reading)

# Convert the corrected data to a DataFrame
corrected_data_df = pd.DataFrame(corrected_data, columns=["Corrected_PM2.5_Readings"])

# Save the DataFrame to a CSV file
corrected_data_df.to_csv("Corrected_Dataset.csv", index=False)

# Import the reference dataset
reference_data = pd.read_csv("Reference1.csv", header=0, index_col=0)

# Convert "date" column to datetime
df_.index = pd.to_datetime(df_.index)

# Data visualisation
plt.figure(figsize=(10, 5))
plt.plot(corrected_data_df, label='Corrected Data', color='green')
plt.plot(reference_data.index, label='Reference Data', color='r')
plt.xlabel("Date", fontsize = 15, fontweight = 'bold')
plt.ylabel("PM2.5 (ug/ m³)", fontsize = 15, fontweight = 'bold')
plt.legend(loc='upper left')

plt.savefig('CorrectedData_RefData_plot.png', dpi=300)
plt.show()

corrected_data_df.head()


# ### ANALYSIS OF MISSING INPUTS
# 
# #### 1- Missing Relative Humidity (RH) Values

# In[30]:


# Condition xgb only over PurpleAir PM2.5 and temperature

grid_search.fit(x_train[["purpleair_pm2.5", "avg_temp"]], x_train[["RefEmb_pm2.5"]])
y_predxgb = grid_search.predict(x_test[["purpleair_pm2.5", "avg_temp"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predxgb)
r2 = r2_score(y_test, y_predxgb)
rmse_value = sqrt(mean_squared_error(y_test, y_predxgb))
bias = n_bias(y_test, y_predxgb)

print('PM2.5c = X1*PAPM2.5 + X2*Temp')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[32]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predxgb, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
legend_fontsize = 'medium'
pol_text = 'y = %.2fx+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('XGBoost Regressor Without RH', fontsize=16, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight='bold')
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight='bold')
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.patch.set_facecolor('white')
ax.set_ylim(0, 80)
ax.set_xlim(0, 80)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('xgboost_regressor_-RH_plot.png', dpi=300)


# #### 2 - Missing Temperature Values

# In[33]:


# Condition xgb with only PurpleAir PM2.5 and RH
grid_search.fit(x_train[["purpleair_pm2.5", "RH"]], x_train[["RefEmb_pm2.5"]])
y_predxgb = grid_search.predict(x_test[["purpleair_pm2.5", "RH"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predxgb)
r2 = r2_score(y_test, y_predxgb)
rmse_value = sqrt(mean_squared_error(y_test, y_predxgb))
bias = n_bias(y_test, y_predxgb)

print('PM2.5c = X1*PAPM2.5 + X2*RH')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[35]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predxgb, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
legend_fontsize = 'medium'
pol_text = 'y = %.2fx+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('XGBoost Regressor Without Temperature', fontsize=15, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight='bold')
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight='bold')
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.patch.set_facecolor('white')
ax.set_ylim(0, 70)
ax.set_xlim(0, 70)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('xgboost_regressor_-Tmp_plot.png', dpi=300)


# #### 3 - Missing Temperature and Relative Humidity (RH) Values

# In[36]:


# Condition xgb with only PurpleAir PM2.5
grid_search.fit(x_train[["purpleair_pm2.5"]], x_train[["RefEmb_pm2.5"]])
y_predxgb = grid_search.predict(x_test[["purpleair_pm2.5"]])

# Evaluating the model  on the test set
mae = mean_absolute_error(y_test, y_predxgb)
r2 = r2_score(y_test, y_predxgb)
rmse_value = sqrt(mean_squared_error(y_test, y_predxgb))
bias = n_bias(y_test, y_predxgb)

print('PM2.5c = X1*PAPM2.5')
print(' R2=  %.3f'  % r2 )
print(' MAE= %.3f '  % mae)
print(' RMSE=  %.3f '  % rmse_value )
print(' bias= %.3f ' % bias)


# In[37]:


# Plotting density scatter graph
ax, fig1, z = density_scatter(y_test.values, y_predxgb, bins=[15, 15], figsize=(8, 6))

# Set the title, labels, and legend
legend_fontsize = 'medium'
pol_text = 'y = %.2fx+%.2f, DB size=%0.1fK' % (z[0], z[1], (dataset_size / 1000))
ax.set_title('XGBoost Regressor Without Temp & RH', fontsize=15, fontweight='bold')
ax.set_xlabel("Observed PM2.5", fontsize=14, fontweight='bold')
ax.set_ylabel("Predicted PM2.5", fontsize=14, fontweight='bold')
ax.legend(['R2 = {:.2f}, MAE= {:.2f}, RMSE= {:.2f}, Bias= {:.2f}'.format(r2, mae, rmse_value, bias), pol_text, 'y = x'],
          fontsize='medium')
ax.tick_params(axis='both', which='major', labelsize=12)
ax.patch.set_facecolor('white')
ax.set_ylim(0, 80)
ax.set_xlim(0, 80)

ax.grid(True, linestyle='-', linewidth=0.25, color='grey')

plt.savefig('xgboost_regressor_-Tmp-RH_plot.png', dpi=300)


# In[ ]:





# In[47]:


# Select the relevant features

df_ = data_h[["purpleair_pm2.5", "avg_temp", "RH", "RefEmb_pm2.5"]]
selected_features = ['purpleair_pm2.5', 'avg_temp', 'RH' ,'RefEmb_pm2.5']
df_selected = df_[selected_features]

# Calculate Spearman correlation matrix
correlation_matrix = df_selected.corr(method='spearman')

# Generate a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Spearman's Correlation Coefficient Heatmap")
plt.show()


# In[ ]:




