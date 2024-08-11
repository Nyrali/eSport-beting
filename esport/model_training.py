
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import RFE
# from mlxtend.feature_selection import FSSearch
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
import json
import os
import pickle 
from scipy import stats
from esport.model_creation_func import evaluate_model, find_best_model, make_feature_importance_df, make_roc_plot, remove_multicollinearity, make_feature_importance_plot
from esport.model_creation_func import make_model_folder, make_summary_json, save_model, make_feature_auc_dict, make_features_histogram, make_features_corrplot, kfold_cross_val_auc, cross_val_auc_plot
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, KFold
import pickle

df = pd.read_csv("D:\\eSport-betting\\esport\\valorant_features_all.csv")

df = pd.read_csv("D:\\eSport-betting\\esport\\valorant_features_one_row_per_match.csv")

df = df.rename(columns={"team_wins":"target"})
df["target"] = df["target"].astype(int)
df = df.dropna()

threshold = 0.95
df= remove_multicollinearity(df, threshold)

X = df.drop(['target'], axis=1) 


X =X[selected_features]

y = df['target']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Create a logistic regression model
model = LogisticRegression(max_iter=1000)
# Train the model
model.fit(X_train, y_train)
# Make predictions on the test set
y_pred = model.predict(X_test)






def make_model_folder(model_file):    # folders making
    
    os.makedirs(f"esport/{model_file}", exist_ok=True)
    os.makedirs(f"esport/{model_file}/features_histograms", exist_ok=True)
    return model_file


model_name = "first_model"
model_file = make_model_folder(model_name)


train_auc_scores, test_auc_scores, auc_diff = kfold_cross_val_auc(model, X, y)
cross_val_auc_plot(test_auc_scores,train_auc_scores, auc_diff, model_file)



metrics_dict = evaluate_model(model, X_train, X_test, y_train, y_test)
roc_plot = make_roc_plot(metrics_dict, model_file=model_file, save=False)
# roc_plot = make_roc_plot(metrics_dict, model_file=model_file, save=True)

feature_importance_df = make_feature_importance_df (X_train, model)
feature_importance_plot = make_feature_importance_plot (feature_importance_df, model_file=model_file, save=False)
summary_json = make_summary_json(df,feature_importance_df, metrics_dict, model_file)


# FOR FINAL MODEL
make_features_histogram (X, model_file)
make_features_corrplot (X, model_file)
save_model (model, feature_importance_df, model_file)



# SELECTING OPTIMAL NUMBER OF FEATURES
# name it according to target
feature_json_name = "first_model"
feature_auc_dict = make_feature_auc_dict (X,y, X_train, X_test, y_train, y_test, feature_json_name)



with open('esport/features_selection_first_model.json', 'r') as f:
    info_dict_loaded = json.load(f)

info_dict_loaded["features_df"] = [pd.DataFrame(data) for data in info_dict_loaded["features_df"]]
auc_diff = [a - b for a, b in zip(info_dict_loaded["train_auc"], info_dict_loaded["test_auc"])]
auc_diff_abs = [abs(i) for i in auc_diff]
info_dict_loaded["auc_diff"] = auc_diff
info_dict_loaded["auc_diff_abs"] = auc_diff_abs

# Feature_AUC plot
plt.figure(figsize=(10, 6))
plt.plot(info_dict_loaded["features_count"], info_dict_loaded["test_auc"], marker='o', color='b', linestyle='-')
plt.plot(info_dict_loaded["features_count"], info_dict_loaded["train_auc"], marker='o', color='r', linestyle='-')
plt.title('Test AUC vs. Number of Features Selected')
plt.xlabel('Number of Features Selected')
plt.ylabel('Test AUC')
plt.grid(True)
plt.show()


# Top results for test auc
top_indices = np.argsort(info_dict_loaded["test_auc"])[-20:][::-1]

top_results = []
for idx in top_indices:
    result_dict = {
        "train_auc": info_dict_loaded["train_auc"][idx],
        "test_auc": info_dict_loaded["test_auc"][idx],
        "auc_diff": info_dict_loaded["auc_diff"][idx],
        "auc_diff_abs": info_dict_loaded["auc_diff_abs"][idx],
        "feature_importance_df": info_dict_loaded["features_df"][idx]
    }
    top_results.append(result_dict)
#
test_auc_values = [result['test_auc'] for result in top_results]

sorted_results = sorted(top_results, key=lambda x: x["test_auc"], reverse=True)
# Print sorted results
for i, result in enumerate(sorted_results):
    print(f"auc_diff_abs: {round(result['auc_diff_abs'], 3)}, test_auc: {round(result['test_auc'], 3)}, feature count: {len(result['feature_importance_df'])}, iloc: {i}")



selected_features = sorted_results[0]["feature_importance_df"]["Feature"].to_list()











def make_feature_auc_dict (X,y, X_train, X_test, y_train, y_test, feature_json_name):

    auc_train_lst = []
    auc_test_lst = []
    feature_importance_df_lst = []
    features_count_lst = []
    runs = range(1,len(X.columns))

    for i in runs:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)    
        model = LogisticRegression(max_iter=1000)   
        model.fit(X_train, y_train)   
        y_pred = model.predict(X_test)
        metrics_dict = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(metrics_dict["roc_auc_train"])
        auc_train_lst.append(metrics_dict["roc_auc_train"])
        auc_test_lst.append(metrics_dict["roc_auc_test"])
        feature_importance_df = make_feature_importance_df (X_train, model)
        feature_importance_df_lst.append(feature_importance_df)    
        feature_importance_df = feature_importance_df.sort_values(by='Abs_coefficient', ascending=False)
        feature_importance_df = feature_importance_df[:-1]
        new_features_lst = feature_importance_df['Feature'].tolist()
        features_count_lst.append(len(new_features_lst))
        X = X[new_features_lst]

        
    info_dict = {"train_auc" : auc_train_lst,
                "test_auc" : auc_test_lst,
                "features_count": features_count_lst,
                "features_df" : feature_importance_df_lst }
   

    info_dict_serializable = {
        "train_auc": info_dict["train_auc"],
        "test_auc": info_dict["test_auc"],
        "features_count": info_dict["features_count"],
        "features_df": [df.to_dict(orient='records') for df in info_dict["features_df"]]
    }

    # Save to JSON file
    with open(f'esport/features_selection_{feature_json_name}.json', 'w') as f:
        json.dump(info_dict_serializable, f, indent=4)

    return info_dict