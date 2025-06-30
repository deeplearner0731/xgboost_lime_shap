import shap
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import warnings
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
warnings.filterwarnings("ignore")
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import re
import random
import itertools
import lime
from lime import lime_tabular
from sklearn.linear_model import Ridge
from collections import defaultdict
##method:A-learning method for continues and binary:
def squared_log_continues(predt,dtrain):
    grad = gradient_continues(predt, dtrain)
    hess = hessian_continues(predt, dtrain)
    return grad, hess
def gradient_continues(predt, dtrain):
    y = dtrain.get_label()
    c=(X_train_trt+1.0)/2.0-pi_train
    return -2.0*c*(y-predt*c)

def hessian_continues(predt, dtrain):
    y = dtrain.get_label()
    c=(X_train_trt+1.0)/2.0-pi_train
    return 2.0*(c**2)
def rmsle_continues(predt, dtrain):
    y = dtrain.get_label()
    y_real=y.reshape((y.shape[0],))
    c=(X_train_trt+1.0)/2.0-pi_train
    elements = (y-c*predt)**2
    acc=(predt-x_train_trt_effect)**2
    print (float(np.sqrt(np.sum(acc) / len(y))))
    
    return 'LOSS', float(np.sqrt(np.sum(elements) / len(y)))

def plot_roc_curve(true_y, y_prob):
    """
    plots the roc curve based of the probabilities
    """
    fpr, tpr, thresholds = roc_curve(true_y, y_prob)
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
##Weight learning:

def squared_log_weight(predt, dtrain):
    """
    Compute gradient and hessian for Weight-learning loss.
    """
    grad = gradient_weight(predt, dtrain)
    hess = hessian_weight(predt, dtrain)
    return grad, hess

def gradient_weight(predt, dtrain):
    """
    Compute the gradient for Weight-learning loss.
    """
    y = dtrain.get_label()
    c = (1.0 - X_train_trt) / 2.0 + pi_train * X_train_trt
    return (-2.0 * X_train_trt) / c * (y - predt * X_train_trt)

def hessian_weight(predt, dtrain):
    """
    Compute the hessian for Weight-learning loss.
    """
    y = dtrain.get_label()
    c = (1.0 - X_train_trt) / 2.0 + pi_train * X_train_trt
    return (2.0 * (X_train_trt**2)) / c

def rmsle_weight(predt, dtrain):
    """
    Compute the Root Mean Squared Logarithmic Error (RMSLE) for Weight-learning.
    """
    y = dtrain.get_label()
    y_real = y.reshape((y.shape[0],))
    c = (1.0 - X_train_trt) / 2.0 + pi_train * X_train_trt
    elements = ((y - X_train_trt * predt)**2) / c
    acc = (predt - x_train_trt_effect)**2
    print(float(np.sqrt(np.sum(acc) / len(y))))
    
    return 'LOSS', float(np.sqrt(np.sum(elements) / len(y)))
    
data = pd.read_csv('/ui/abv/liuzx18/project_shapley/simulation_data/beta0_100sims.csv')


best_params={'learning_rate':0.005,
     'verbosity':1,
     'booster':'gbtree',
     'max_depth':1,
     'lambda':5,
     'tree_method':'hist' 
    }
best_epoch=1000
num_permutations=100
features = [f"x{i}" for i in range(1, 11)]
all_p = pd.DataFrame(index=features)
train_all=[]
test_all=[]
p_all=[]

for sim_id in range(1, 101):
    data_simulation = data[data['sim_id'] == sim_id]
    data_simulation_test = data[data['sim_id'] == sim_id]
    x_df=data_simulation[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
    X_=np.array(x_df)[:]
    x_df_test=data_simulation_test[['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10']]
    X_test=np.array(x_df_test)
    y_train=np.array(data_simulation[['y']])
    y_train=y_train.reshape(1000,)
    y_test=np.array(data_simulation_test[['y']])
    y_test=y_test.reshape(1000,)
    trt_=data_simulation[['treatment']]
    g_real=data_simulation[['sigpos']]
    g_real_test=data_simulation_test[['sigpos']]
    logreg = LogisticRegression()
    logreg.fit(X_,trt_)
    pi_x = logreg.predict_proba(X_)
    pi_train=pi_x[:,1]
    X_train_trt=np.where(trt_==1,1,-1)
    X_train_trt=X_train_trt.reshape(1000,)
    dtrain = xgb.DMatrix(X_, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    x_train_feature_pd=x_df
    # Train the model using the best parameters
    method = 'w'
    if method == "a":
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_epoch,  # Use the best num_boost_round
            obj=squared_log_continues,
            feval=rmsle_continues
        )
    elif method =='w':
         model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_epoch,  # Use the best num_boost_round
            obj=squared_log_weight,
            feval=rmsle_weight
        )

    # Predictions and Evaluation on Train Set
    pred_train = model.predict(dtrain)
    pred_train_prob = 1.0 / (1.0 + np.exp(-pred_train))  # Sigmoid transformation if needed
    auc_train = roc_auc_score(g_real.astype(int), pred_train_prob)
    #print(f"Final Train AUC: {auc_train}")
    train_all.append(auc_train)

    # Predictions and Evaluation on Test Set
    pred_test = model.predict(dtest)
    pred_test_prob = 1.0 / (1.0 + np.exp(-pred_test))  # Sigmoid transformation if needed
    auc_test = roc_auc_score(g_real_test.astype(int), pred_test_prob)
    #print(f"Final Test AUC: {auc_test}")
    test_all.append(auc_test)
    # Assuming 'model' and 'x_train_feature_pd' are already defined and set up
    f_importance = model.get_score(importance_type='gain')
    f_importance = {k: v for k, v in sorted(f_importance.items(), key=lambda item: item[1])}
    importance_df = pd.DataFrame.from_dict(data=f_importance, orient='index')
    pd_colnames = []
    for j in importance_df.index:   
        r = re.findall(r'\d+', j)    
        pd_colnames.append(int(r[0]))    
    importance_df.index = x_train_feature_pd.columns[pd_colnames]
    importance_df.columns=['Importance']

    all_features = [f'x{i}' for i in range(1,X_.shape[1]+1)]
    # Assume original_importance is a DataFrame with index as feature names and a column named 'Importance'
    original_importance = importance_df.copy()
    original_importance.columns = ['Importance']
    original_importance_dict = {f: original_importance.loc[f, 'Importance'] if f in original_importance.index else 0.0 for f in all_features}
    T_original = max(abs(v) for v in original_importance_dict.values())
    
    
    
    T_perm_list = []
    perm_importance_dicts = []

    for _ in range(num_permutations):

        trt_=np.array(data_simulation[['treatment']])
        np.random.shuffle(trt_)
        logreg = LogisticRegression()
        logreg.fit(X_,trt_)
        pi_x = logreg.predict_proba(X_)
        pi_train=pi_x[:,1]
        X_train_trt=np.where(trt_==1,1,-1)
        X_train_trt=X_train_trt.reshape(1000,)
        model_perm = xgb.train(
            best_params,
            dtrain,
            num_boost_round=best_epoch,  # Use the best num_boost_round
            obj=squared_log_weight,
            feval=rmsle_weight
        )
        perm_importance = model_perm.get_score(importance_type='gain')
        f_importance = {k: v for k, v in sorted(perm_importance.items(), key=lambda item: item[1])}
        perm_importance_df  = pd.DataFrame.from_dict(data=f_importance, orient='index')
        perm_importance_df.columns = ['Importance']
        rename_dict = {f'f{i}': f'x{i+1}' for i in range(10)}
        perm_importance_df.rename(index=rename_dict, inplace=True)
        perm_complete= {f: perm_importance_df.loc[f, 'Importance'] if f in perm_importance_df.index else 0.0 for f in all_features}
        T_perm_list.append(max(abs(v) for v in perm_complete.values()))

        perm_importance_dicts.append(perm_complete)

    # Global max-type p-value
    p_global = (1 + sum(T >= T_original for T in T_perm_list)) / (num_permutations + 1)

    p_all.append(p_global)



