# importing necessary Python libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
#import plotly.offline as pyoff
import plotly.graph_objs as go 
#import plotly.figure_factory as ff

# avoid displaying warnings
import warnings
warnings.filterwarnings("ignore")

#import machine learning related libraries
from sklearn.svm import SVC
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score, train_test_split, GridSearchCV, cross_validate
from multiscorer import MultiScorer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.cluster import KMeans
import xgboost as xgb
import time 

# Loading the data
df = pd.read_csv('online_retail_II.csv')
# Rename the following columns: 
#    Invoice to InvoiceNo
#    Customer ID to CustomerID
#    Price to UnitPrice

df.rename(columns={'Invoice':'InvoiceNo', 'Customer ID':'CustomerID', 
                   'Price':'UnitPrice'}, 
          inplace=True)

# Drop the missing data in `df`
df_data = df.dropna()

# Convert the date field, InvoiceDate to datetime object
df_data.InvoiceDate = pd.to_datetime(df_data.InvoiceDate)
ctm_bhvr_dt = df_data[(df_data.InvoiceDate < pd.Timestamp(2011,9,1)) & (df_data.InvoiceDate >= pd.Timestamp(2009,12,1))].reset_index(drop=True)
ctm_next_quarter = df_data[(df_data.InvoiceDate < pd.Timestamp(2011,12,1)) & (df_data.InvoiceDate >= pd.Timestamp(2011,9,1))].reset_index(drop=True)
# Get the distinct customers in the dataframe ctm_bhvr_dt
ctm_dt = pd.DataFrame(ctm_bhvr_dt['CustomerID'].unique())

# Rename the column to CustomerID.
ctm_dt.columns = ['CustomerID']
# Create a dataframe with CustomerID and customers first purchase 
# date in the dataset ctm_next_quarter
ctm_1st_purchase_in_next_quarter = ctm_next_quarter.groupby('CustomerID').InvoiceDate.min().reset_index()
ctm_1st_purchase_in_next_quarter.columns = ['CustomerID', 'MinPurchaseDate']

# Create a dataframe with CustomerID and customers last purchase 
# date in the dataset ctm_bhvr_dt
ctm_last_purchase_bhvr_dt = ctm_bhvr_dt.groupby('CustomerID').InvoiceDate.max().reset_index()
ctm_last_purchase_bhvr_dt.columns = ['CustomerID', 'MaxPurchaseDate']

# Merge two dataframes ctm_last_purchase_bhvr_dt and ctm_1st_purchase_in_next_quarter
ctm_purchase_dates = pd.merge(ctm_last_purchase_bhvr_dt, ctm_1st_purchase_in_next_quarter, on='CustomerID', how='left')

# Get the difference in days from MinPurchaseDate and MaxPurchaseDate for each customer
ctm_purchase_dates['NextPurchaseDay'] = (ctm_purchase_dates['MinPurchaseDate'] - ctm_purchase_dates['MaxPurchaseDate']).dt.days

# Update the dataframe ctm_dt by merging it with the NextPurchaseDay column of the dataframe ctm_purchase_dates
ctm_dt = pd.merge(ctm_dt, ctm_purchase_dates[['CustomerID', 'NextPurchaseDay']], on='CustomerID', how='left')
# Fill all missing values in the dataset ctm_dt with the number 9999
ctm_dt = ctm_dt.fillna(9999)
ctm_dt.head()
# Get the maximum purchase date of each customer and create a dataframe with it together with the customer's id.
ctm_max_purchase = ctm_bhvr_dt.groupby('CustomerID').InvoiceDate.max().reset_index()
ctm_max_purchase.columns = ['CustomerID','MaxPurchaseDate']

# Find the recency of each customer in days
ctm_max_purchase['Recency'] = (ctm_max_purchase['MaxPurchaseDate'].max() - ctm_max_purchase['MaxPurchaseDate']).dt.days

# Merge the dataframes ctm_dt and ctm_max_purchase[['CustomerID', 'Recency']] on the CustomerID column.
ctm_dt = pd.merge(ctm_dt, ctm_max_purchase[['CustomerID', 'Recency']], on='CustomerID')
kmeans = KMeans(n_clusters=4)
kmeans.fit(ctm_dt[['Recency']])
ctm_dt['RecencyCluster'] = kmeans.predict(ctm_dt[['Recency']])
ctm_dt = order_cluster(ctm_dt, 'Recency', 'RecencyCluster', False)
ctm_dt.head()
# Get the descriptive statistics of the Recency data of each  RecencyCluster value

ctm_dt.groupby('RecencyCluster')['Recency'].describe()
# Get order counts for each user and create a dataframe with it
ctm_frequency = df_data.groupby('CustomerID').InvoiceDate.count().reset_index()
ctm_frequency.columns = ['CustomerID', 'Frequency']

# Update the dataframe ctm_dt by merging it with the dataframe ctm_frequency
ctm_dt = pd.merge(ctm_dt, ctm_frequency, on='CustomerID')
ctm_dt.head()
# Apply clustering to customer frequency data
kmeans = KMeans(n_clusters=4)
kmeans.fit(ctm_dt[['Frequency']])
ctm_dt['FrequencyCluster'] = kmeans.predict(ctm_dt[['Frequency']])

# Order the dataframe based on the cluster values in FrequencyCluster
ctm_dt = order_cluster(ctm_dt, 'Frequency', 'FrequencyCluster', False)
ctm_dt.head()
# Get the descriptive statistics of the Frequency data of each  FrequencyCluster value

ctm_dt.groupby('FrequencyCluster')['Frequency'].describe()
# Create a new label, Revenue of each item bought
df_data['Revenue'] = df_data.UnitPrice * df_data.Quantity

# Get the revenue from each customer and sum them.
ctm_revenue = df_data.groupby('CustomerID').Revenue.sum().reset_index()

# Merge the dataframe ctm_revenue with ctm_dt
ctm_dt = pd.merge(ctm_dt, ctm_revenue, on='CustomerID')
ctm_dt.head()
# Apply clustering to customer revenue data
kmeans = KMeans(n_clusters=number_of_clusters)
kmeans.fit(ctm_dt[['Revenue']])
ctm_dt['RevenueCluster'] = kmeans.predict(ctm_dt[['Revenue']])

# Order the dataframe based on the cluster values in RevenueCluster
ctm_dt = order_cluster(ctm_dt, 'Revenue', 'RevenueCluster', True)

# Group the dataframe ctm_dt by RevenueCluster and get the statistical description of the Revenue data of each of cluster value
ctm_dt.groupby('RevenueCluster')['Revenue'].describe()
# Calculate overall score 
ctm_dt['OverallScore'] = ctm_dt['RecencyCluster'] + ctm_dt['FrequencyCluster'] + ctm_dt['RevenueCluster']

# Get mean of the components of OverallScore
ctm_dt.groupby('OverallScore')['Recency', 'Frequency', 'Revenue'].mean()
# Categories customers' into Segement based on their OverallScore
ctm_dt['Segment'] = 'Low-Value'

ctm_dt.loc[ctm_dt['OverallScore'] > 4, 'Segment'] = 'Mid-Value'

ctm_dt.loc[ctm_dt['OverallScore'] > 6, 'Segment'] = 'High-Value'
ctm_dt.head()
# Create a copy the dataframe ctm_dt
ctm_class = ctm_dt.copy()

# Convert categorical data to numerical data using get_dummies
ctm_class = pd.get_dummies(ctm_class)
# Set the column 'NextPurchaseDayRange' to 1. Note that this  
# captures customers with NextPurchaseDay value is less than 90 days

ctm_class['NextPurchaseDayRange'] = 1 

# Set the value of NextPurchaseDayRange to 0 for customers whose NextPurchaseDay is more than 90 days.
ctm_class.loc[ctm_class.NextPurchaseDay>90, 'NextPurchaseDayRange'] = 0
# Calculating correlation matrix
corr_matrix = ctm_class[ctm_class.columns].corr()

# Create a dataframe for the minimum correlation coefficient values
corr_df = pd.DataFrame(corr_matrix.min())

# Rename the column label of corr_df
corr_df.columns = ['MinCorrelationCoeff']

# Calculate the maximum correlation coefficient value apart less than 1.
corr_df['MaxCorrelationCoeff'] = corr_matrix[corr_matrix < 1].max()
corr_df
plt.figure(figsize = (40, 30))
sns.heatmap(corr_matrix, annot = True, linewidths=0.2, 
            fmt=".2f");
ctm_class = ctm_class.drop('NextPurchaseDay', axis=1)
X, y = ctm_class.drop('NextPurchaseDayRange', axis=1), ctm_class.NextPurchaseDayRange

# Split feature data X and target data y into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True)

# Create an array of models
models = []
models.append(("LogisticRegression", LogisticRegression()))
models.append(("GaussianNB", GaussianNB()))
models.append(("RandomForestClassifier", RandomForestClassifier()))
models.append(("SVC", SVC()))
models.append(("DecisionTreeClassifier", DecisionTreeClassifier()))
models.append(("xgb.XGBClassifier", xgb.XGBClassifier(eval_metric='mlogloss')))
models.append(("KNeighborsClassifier", KNeighborsClassifier()))

# Measuring the metrics of the different models
scorer = MultiScorer({'accuracy'  : (accuracy_score , {}), 
                      'f1_score'  : (f1_score       , {'pos_label': 3, 'average':'macro'}), 
                      'recall'    : (recall_score   , {'pos_label': 3, 'average':'macro'}), 
                      'precision' : (precision_score, {'pos_label': 3, 'average':'macro'})
                     })
# A dictionary for all the distinct models and their respective metrics
model_scores_dict = {'model_name' : [], 
                     'accuracy'   : [], 
                     'f1_score'   : [], 
                     'recall'     : [], 
                     'precision'  : [],
                     'time'       : []
                    }
# For each model name and model in models
for model_name, model in models: 
    
    # Add model_name to model_scores_dict 
    model_scores_dict['model_name'].append(model_name)
    #print(model_name)
    kfold = KFold(n_splits=2, random_state=24, shuffle=True)
    start = time.time()
    _ = cross_val_score(model, X_train, y_train, cv = kfold, scoring = scorer)
    cv_result = scorer.get_results()
    
    # For each metric in cv_result.keys()
    for metric_name in cv_result.keys():
        # Get the average of cv_result[metric_name]
        average_score = np.average(cv_result[metric_name])
        # Update model_scores_dict with average_score for model_name
        model_scores_dict[metric_name].append(average_score)
        #print('%s : %f' %(metric_name, average_score))
model_scores_dict['time'].append((time.time() - start))
    #print('time : ', time.time() - start, '\n\n')
        
model_score_df = pd.DataFrame(model_scores_dict).set_index("model_name")
model_score_df.sort_values(by=["accuracy", "f1_score", "time"], ascending=False)
parameter = {
    'max_depth':range(3,10,2), 
    'min_child_weight':range(1,5,2)
    }

p_grid_search = GridSearchCV(estimator = xgb.XGBClassifier(eval_metric='mlogloss'), 
                             param_grid = parameter, 
                             scoring='accuracy', 
                             n_jobs=-1, 
                             cv=2
                            )

p_grid_search.fit(X_train, y_train)

p_grid_search.best_params_, p_grid_search.best_score_
refined_xgb_model = xgb.XGBClassifier(eval_metric='logloss', 
                                      max_depth=list(p_grid_search.best_params_.values())[0]-1, 
                                      min_child_weight=list(p_grid_search.best_params_.values())[-1]+4
                                     ).fit(X_train, y_train)

print('Accuracy of refined XGB classifier on training set: {:.2f}'.format(refined_xgb_model.score(X_train, y_train)))
print('Accuracy of refined XGB classifier on test set: {:.2f}'.format(refined_xgb_model.score(X_test[X_train.columns], y_test)))
# Predict the target values for the X_test with the refined XGB-Classifier
ref_xgb_pred_y = refined_xgb_model.predict(X_test)

# Predict the target values for the X_test with the Logistic Regression-Classifier
log_reg_pred_y = LogisticRegression().fit(X_train, y_train).predict(X_test)


# A dictionary of model names with the various metrics
ref_xgb_log_reg_dict = {"model_name" : ["xgb.XGBClassifier", "LogisticRegression"], 
                        "accuracy"   : [accuracy_score(y_test, ref_xgb_pred_y), accuracy_score(y_test, log_reg_pred_y)], 
                        "f1_score"   : [f1_score(y_test, ref_xgb_pred_y), f1_score(y_test, log_reg_pred_y)], 
                        "recall"     : [recall_score(y_test, ref_xgb_pred_y), recall_score(y_test, log_reg_pred_y)], 
                        "precision"  : [precision_score(y_test, ref_xgb_pred_y), precision_score(y_test, log_reg_pred_y)]
                       }

# Create a dataframe with ref_xgb_log_reg_dict
ref_xgb_log_reg_df = pd.DataFrame(ref_xgb_log_reg_dict).set_index("model_name")

# Order the dataframe ref_xgb_log_reg_df by the metric values in increasing order
ref_xgb_log_reg_df.sort_values(by=["accuracy", "f1_score", "recall", "precision"], ascending=False)