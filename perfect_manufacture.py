import pandas as pd
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from scipy.stats import mode
import numpy as np
value1_df = pd.read_csv("./경진대회 데이터셋/train_input/train_input_스핀들전력.csv")
value2_df = pd.read_csv("./경진대회 데이터셋/train_input/train_input_스핀들진동.csv")
value3_df = pd.read_csv("./경진대회 데이터셋/train_input/train_input_전체전력.csv")
target_df = pd.read_csv("./경진대회 데이터셋/train_output.csv") 
tool_df = target_df.iloc[:,0]
condition_df = target_df.iloc[:,1]
tool_df.to_csv("./경진대회 데이터셋/tool_df.csv",index=False)
condition_df.to_csv("./경진대회 데이터셋/condition_df.csv",index=False) 


mean_data,max_data,min_data,sum_data,var_data,std_data,\
median_data,mode_data,power_peak_mean_data,power_peak_max_data,\
power_peak_min_data,power_peak_sum_data,power_peak_var_data,\
power_peak_std_data,power_peak_median_data,power_peak_len_data = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

for i in range(0,410):  
    power = value1_df.iloc[i,:]
    peaks, _ = find_peaks(value1_df.iloc[i,:])
    power_peak = value1_df.iloc[i,peaks]
    mean_data.append(power.mean())
    max_data.append(power.max())
    min_data.append(power.min())
    sum_data.append(power.sum())
    var_data.append(power.var())
    std_data.append(power.std())
    median_data.append(power.median())
    mode_data.append(mode(power))
    power_peak_mean_data.append(power_peak.mean())
    power_peak_max_data.append(power_peak.max())
    power_peak_min_data.append(power_peak.min())
    power_peak_sum_data.append(power_peak.sum())
    power_peak_var_data.append(power_peak.var())
    power_peak_std_data.append(power_peak.std())
    power_peak_median_data.append(power_peak.median())  
    power_peak_len_data.append(len(peaks))  
    
power_mean_df = pd.DataFrame(mean_data)
power_max_df = pd.DataFrame(max_data)
power_min_df = pd.DataFrame(min_data)
power_sum_df = pd.DataFrame(sum_data)
power_var_df = pd.DataFrame(var_data)
power_std_df = pd.DataFrame(std_data)
power_median_df = pd.DataFrame(median_data)
power_mode_df = pd.DataFrame(mode_data)

power_peak_mean_df = pd.DataFrame(power_peak_mean_data)
power_peak_mean_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_max_df = pd.DataFrame(power_peak_max_data)
power_peak_max_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_min_df = pd.DataFrame(power_peak_min_data)
power_peak_min_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_sum_df = pd.DataFrame(power_peak_sum_data)

power_peak_var_df = pd.DataFrame(power_peak_var_data)
power_peak_var_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_std_df = pd.DataFrame(power_peak_std_data)
power_peak_std_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_median_df = pd.DataFrame(power_peak_median_data)
power_peak_median_df.fillna(value1_df.iloc[i,peaks].median(),inplace=True)

power_peak_len_df = pd.DataFrame(power_peak_len_data)

vib_mean_data,vib_max_data,vib_min_data,vib_sum_data,vib_var_data,vib_std_data,\
vib_median_data,vib_mode_data,vib_peak_mean_data,vib_peak_max_data,\
vib_peak_min_data,vib_peak_sum_data,vib_peak_var_data,vib_peak_std_data,\
vib_peak_median_data,vib_peak_len_data = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

for i in range(0,410):  
    vib = value2_df.iloc[i,:]
    peaks, _ = find_peaks(value2_df.iloc[i,:])
    vib_peak = value2_df.iloc[i,peaks]
    vib_mean_data.append(vib.mean())
    vib_max_data.append(vib.max())
    vib_min_data.append(vib.min())
    vib_sum_data.append(vib.sum())
    vib_var_data.append(vib.var())
    vib_std_data.append(vib.std())
    vib_median_data.append(vib.median())
    vib_mode_data.append(mode(vib))
    vib_peak_mean_data.append(vib_peak.mean())
    vib_peak_max_data.append(vib_peak.max())
    vib_peak_min_data.append(vib_peak.min())
    vib_peak_sum_data.append(vib_peak.sum())
    vib_peak_var_data.append(vib_peak.var())
    vib_peak_std_data.append(vib_peak.std())
    vib_peak_median_data.append(vib_peak.median()) 
    vib_peak_len_data.append(len(peaks))
    
vib_mean_df = pd.DataFrame(vib_mean_data)
vib_max_df = pd.DataFrame(vib_max_data)
vib_min_df = pd.DataFrame(vib_min_data)
vib_sum_df = pd.DataFrame(vib_sum_data)
vib_var_df = pd.DataFrame(vib_var_data)
vib_std_df = pd.DataFrame(vib_std_data)
vib_median_df = pd.DataFrame(vib_median_data)
vib_mode_df = pd.DataFrame(vib_mode_data)    

vib_peak_mean_df = pd.DataFrame(vib_peak_mean_data)
vib_peak_mean_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_max_df = pd.DataFrame(vib_peak_max_data)
vib_peak_max_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_min_df = pd.DataFrame(vib_peak_min_data)
vib_peak_min_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_sum_df = pd.DataFrame(vib_peak_sum_data)

vib_peak_var_df = pd.DataFrame(vib_peak_var_data)
vib_peak_var_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_std_df = pd.DataFrame(vib_peak_std_data)
vib_peak_std_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_median_df = pd.DataFrame(vib_peak_median_data)
vib_peak_median_df.fillna(value2_df.iloc[i,peaks].median(),inplace=True)

vib_peak_len_df = pd.DataFrame(vib_peak_len_data)

all_power_mean_data,all_power_max_data,all_power_min_data,all_power_sum_data,\
all_power_var_data,all_power_std_data,all_power_median_data,all_power_mode_data,\
all_power_peak_mean_data,all_power_peak_max_data,all_power_peak_min_data,\
all_power_peak_sum_data,all_power_peak_var_data,all_power_peak_std_data,\
all_power_peak_median_data,all_power_peak_len_data = [],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

for i in range(0,410):  
    all_power = value3_df.iloc[i,:]
    peaks, _ = find_peaks(value3_df.iloc[i,:])
    all_power_peak = value3_df.iloc[i,peaks]
    all_power_mean_data.append(all_power.mean())
    all_power_max_data.append(all_power.max())
    all_power_min_data.append(all_power.min())
    all_power_sum_data.append(all_power.sum())
    all_power_var_data.append(all_power.var())
    all_power_std_data.append(all_power.std())
    all_power_median_data.append(all_power.median())
    all_power_mode_data.append(mode(all_power))
    all_power_peak_mean_data.append(all_power_peak.mean())
    all_power_peak_max_data.append(all_power_peak.max())
    all_power_peak_min_data.append(all_power_peak.min())
    all_power_peak_sum_data.append(all_power_peak.sum())
    all_power_peak_var_data.append(all_power_peak.var())
    all_power_peak_std_data.append(all_power_peak.std())
    all_power_peak_median_data.append(all_power_peak.median()) 
    all_power_peak_len_data.append(len(peaks))
    
all_power_mean_df = pd.DataFrame(all_power_mean_data)
all_power_max_df = pd.DataFrame(all_power_max_data)
all_power_min_df = pd.DataFrame(all_power_min_data)
all_power_sum_df = pd.DataFrame(all_power_sum_data)
all_power_var_df = pd.DataFrame(all_power_var_data)
all_power_std_df = pd.DataFrame(all_power_std_data)
all_power_median_df = pd.DataFrame(all_power_median_data)
all_power_mode_df = pd.DataFrame(all_power_mode_data)    

all_power_peak_mean_df = pd.DataFrame(all_power_peak_mean_data)
all_power_peak_max_df = pd.DataFrame(all_power_peak_max_data)
all_power_peak_min_df = pd.DataFrame(all_power_peak_min_data)
all_power_peak_sum_df = pd.DataFrame(all_power_peak_sum_data)
all_power_peak_var_df = pd.DataFrame(all_power_peak_var_data)
all_power_peak_std_df = pd.DataFrame(all_power_peak_std_data)
all_power_peak_median_df = pd.DataFrame(all_power_peak_median_data)
all_power_peak_len_df = pd.DataFrame(all_power_peak_len_data)

vib_mode_df['mode'] = vib_mode_df['mode'].str[0]
vib_mode_df['count'] = vib_mode_df['count'].str[0]

all_power_mode_df['mode'] = all_power_mode_df['mode'].str[0]
all_power_mode_df['count'] = all_power_mode_df['count'].str[0]

all_data_concat_df = []
all_data_concat_df = pd.concat([
                               vib_mean_df,vib_sum_df,
                               all_power_peak_median_df,vib_peak_max_df,
                               vib_peak_min_df,vib_peak_mean_df,
                               all_power_min_df,vib_peak_sum_df,
                               vib_mode_df,vib_median_df,
                               vib_max_df,all_power_mode_df,
                               all_power_median_df,power_peak_min_df,vib_peak_len_df,
                               power_peak_mean_df,all_power_peak_max_df,
                               vib_min_df,vib_peak_var_df], axis=1)
print(all_data_concat_df.isnull().sum())


X = all_data_concat_df
y = condition_df

from sklearn.model_selection import train_test_split    
X_train,X_test,y_train,y_test = \
            train_test_split(X,y,random_state=42,
                              test_size=0.2,stratify=y)


from sklearn.model_selection import StratifiedKFold   
from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import TimeSeriesSplit

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)


param_grid={'n_estimators' : [2,3,4,5,6,7,8],
           'max_depth' : [2,3,4,5,6,7,8,9],
           'min_samples_leaf' : [2,4,6,8,12,18],
           'min_samples_split' : [2,4,6,8,16,20]}

kfold = StratifiedKFold(n_splits=10)

grid_cv = GridSearchCV(rf, param_grid, 
                       cv = kfold, scoring = 'f1', 
                       refit = True, n_jobs = -1)

grid_cv.fit(X_train, y_train)

print('best validation score: %.3f' \
      %grid_cv.best_score_)
    
print(grid_cv.best_params_)


bestModel = grid_cv.best_estimator_

from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import confusion_matrix

y_test_pred = bestModel.predict(X_test)
y_test_prob = bestModel.predict_proba(X_test)

print("test data Confusion matrix:")
print(confusion_matrix(y_test, y_test_pred))

print("test data accuracy: %.3f" \
      %accuracy_score(y_test, y_test_pred))
print("test data f1 score: %.3f" \
      %f1_score(y_test, y_test_pred))


