import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import os
import pickle 
import json
import seaborn as sns
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import roc_auc_score

def remove_multicollinearity(df, threshold):
    # Define the threshold for high correlation
    #threshold = 0.95
    correlation_matrix = df.corr()
    # Identify pairs of highly correlated features
    high_correlation_pairs = [(i, j) for i in correlation_matrix.columns if i != 'target'
                            for j in correlation_matrix.columns if j != 'target'
                            and i != j and abs(correlation_matrix.loc[i, j]) > threshold]

    # Determine which features to drop
    features_to_drop = set()
    for i, j in high_correlation_pairs:
        if abs(correlation_matrix.loc[i, 'target']) >= abs(correlation_matrix.loc[j, 'target']):
            features_to_drop.add(j)
        else:
            features_to_drop.add(i)

    # Drop the features
    df = df.drop(columns=features_to_drop)
    return df


def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Predict classes for the testing set
    y_pred = model.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    # Generate classification report
    classification_rep = classification_report(y_test, y_pred)
    
    # Predict probabilities for the training and testing sets
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Separate the predicted probabilities for the train set
    proba_positive_train = y_train_prob[y_train == 1]
    proba_negative_train = y_train_prob[y_train == 0]
    
    # Separate the predicted probabilities for the test set
    proba_positive_test = y_test_prob[y_test == 1]
    proba_negative_test = y_test_prob[y_test == 0]
    
    # Apply the K-S test train
    ks_statistic_train, ks_p_value_train = stats.ks_2samp(proba_positive_train, proba_negative_train)
    
    # Apply the K-S test test
    ks_statistic_test, ks_p_value_test = stats.ks_2samp(proba_positive_test, proba_negative_test)
    
    # Compute ROC curve and AUC for training set
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_prob)
    roc_auc_train = auc(fpr_train, tpr_train)
    
    # Compute ROC curve and AUC for testing set
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_prob)
    roc_auc_test = auc(fpr_test, tpr_test)

    # Compute Gini
    gini_train = (2*roc_auc_train)-1
    gini_test = (2*roc_auc_test)-1
    
    metrics =  {
        'accuracy': accuracy,
        'confusion_matrix': conf_matrix,
        'classification_report': classification_rep,
        'ks_statistic_train': ks_statistic_train,
        'ks_p_value_train': ks_p_value_train,
        'ks_statistic_test': ks_statistic_test,
        'ks_p_value_test': ks_p_value_test,
        'roc_auc_train': roc_auc_train,
        'roc_auc_test': roc_auc_test,
        'fpr_train': fpr_train,
        'tpr_train': tpr_train,
        'fpr_test': fpr_test,
        'tpr_test': tpr_test,
        "gini_train": gini_train,
        "gini_test": gini_test
    }

    return metrics

def find_best_model(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2', 'elasticnet', 'none'],
        'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
    }

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=1, n_jobs=-1)
    # Fit the model
    grid_search.fit(X_train, y_train)
    # Best hyperparameters
    # Best model
    best_model = grid_search.best_estimator_
    # Predictions
    y_pred = best_model.predict(X_test)


    return y_pred, best_model



def make_roc_plot(metrics_dict, model_file='default_model', save=False):
    plt.figure(figsize=(8, 6))
    plt.plot(metrics_dict["fpr_train"], metrics_dict["tpr_train"], color='darkorange', lw=2, label=f'Training ROC curve (AUC = {metrics_dict["roc_auc_train"]:.2f})')
    plt.plot(metrics_dict["fpr_test"], metrics_dict["tpr_test"], color='green', lw=2, label=f'Testing ROC curve (AUC = {metrics_dict["roc_auc_test"]:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")

    if save:
        directory = f"esport/{model_file}"
        if not os.path.exists(directory):
            os.makedirs(directory)
        plt.savefig(f"{directory}/roc_auc.png")
    
    plt.show()
    plt.clf()
   
    return plt.show()

def make_feature_importance_df (X_train, model):
    # Get feature names
    feature_names = X_train.columns
    # Get the coefficients (feature importances)
    coefficients = model.coef_[0]
    # Take the absolute values of coefficients
    absolute_coefficients = np.abs(coefficients)
    # Create a DataFrame to organize feature names and their corresponding absolute coefficients
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients, "Abs_coefficient": absolute_coefficients})
    # Sort the DataFrame by absolute coefficient values for better visualization
    feature_importance_df = feature_importance_df.sort_values(by='Coefficient', ascending=False)

    return feature_importance_df

def make_feature_importance_plot (feature_importance_df, model_file='default_model',save=False):

    # Plot feature importance with absolute values
    plt.figure(figsize=(15, 8))
    plt.barh(range(len(feature_importance_df)), feature_importance_df['Coefficient'], color='skyblue')
    plt.yticks(range(len(feature_importance_df)), feature_importance_df['Feature'])
    plt.xlabel('Coefficient Value')
    plt.title('Logistic Regression Feature Importance')
    
    if save:
            directory = f"esport/{model_file}"
            if not os.path.exists(directory):
                os.makedirs(directory)
            plt.savefig(f"{directory}/feature_importance.png")

    
    plt.show()
    plt.clf()

    return plt.show()


from sklearn.model_selection import learning_curve

def plot_learning_curves(model, X_train, y_train, X_test, y_test):
    train_sizes, train_scores, test_scores = learning_curve(model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy')

    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_scores_mean, 'o-', color='r', label='Training accuracy')
    plt.plot(train_sizes, test_scores_mean, 'o-', color='g', label='Test accuracy')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()



def make_summary_json(df,feature_importance_df, metrics_dict, model_file):

    features = [i for i in feature_importance_df["Feature"]]
    coefficients = [i for i in feature_importance_df["Coefficient"]]

    info_dict = {
        "Dataset_info": {
            "observations_count": len(df),
            "defaults_count": len(df[df["target"] == 1]),
            
        },
        "Metrics": {
            "auc_train": metrics_dict['roc_auc_train'],
            "auc_test": metrics_dict['roc_auc_test'],
            "gini_train": metrics_dict['gini_train'],
            "gini_test": metrics_dict['gini_test'],
            "accuracy_score": metrics_dict['accuracy'],
            "ks_statistic_train": metrics_dict['ks_statistic_train'],
            "ks_p_value_train": metrics_dict['ks_p_value_train'],
            "ks_statistic_test": metrics_dict['ks_statistic_test'],
            "ks_p_value_test": metrics_dict['ks_p_value_test']

        },
        "Feature_importance": dict(zip(features, coefficients)),
    }

    # Write the info_dict to a JSON file
    with open(f"esport/{model_file}/model_info.json", "w") as json_file:
        json.dump(info_dict, json_file, indent=4)

    return info_dict



def make_model_folder(model_file):    # folders making
    
    os.makedirs(f"esport/{model_file}", exist_ok=True)
    os.makedirs(f"esport/{model_file}/features_histograms", exist_ok=True)
    return model_file


def save_model (model, feature_importance_df, model_file):

    feature_names = [i for i in feature_importance_df["Feature"]]    

    # Save the model with feature names
    model_with_names = {
        'model': model,
        'feature_names': feature_names
    }

    with open(f"esport/{model_file}/{model_file}.pkl", 'wb') as file:
        pickle.dump(model_with_names, file)





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


def make_features_histogram (X, model_file):

    # Features histograms
    for i in X:
        sns.set_style("whitegrid")  # You can choose a different style as neede
        # bin_df[bin_df["revenue_sum"].between(2000000,10000000)]["revenue_sum"].plot(kind="hist", bins=100, edgecolor="k")
        X[i].plot(kind="hist", bins=50, edgecolor="k")
        plt.title(f"Histogram of {i}")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig(f"esport/{model_file}/features_histograms/{i}.png")
        plt.clf()

def make_features_corrplot (X, model_file):
    # Corr plot
    correlation_matrix = X.corr()
    # Create the correlation plot using seaborn's heatmap
    plt.figure(figsize=(15, 10))  # Set the size of the plot
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Plot')
    plt.savefig(f"esport/{model_file}/corr_plot.png")    
    plt.show()
    plt.clf()



def kfold_cross_val_auc(model, X, y):

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    train_auc_scores = []
    test_auc_scores = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model.fit(X_train, y_train)

        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        y_test_pred_prob = model.predict_proba(X_test)[:, 1]

        train_auc = roc_auc_score(y_train, y_train_pred_prob)
        test_auc = roc_auc_score(y_test, y_test_pred_prob)

        train_auc_scores.append(train_auc)
        test_auc_scores.append(test_auc)
        
    train_auc_scores = np.array(train_auc_scores)
    test_auc_scores = np.array(test_auc_scores)
    auc_diff = train_auc_scores - test_auc_scores


    return train_auc_scores, test_auc_scores, auc_diff



def cross_val_auc_plot(test_auc_scores, train_auc_scores, auc_diff, model_file):
    plt.figure(figsize=(10, 6))
    plt.plot(test_auc_scores, marker='o', color='b', linestyle='-', label='Test AUC')
    plt.plot(train_auc_scores, marker='o', color='r', linestyle='-', label='Train AUC')
    plt.title('Test AUC vs. Train AUC - Cross validation')
    plt.ylabel('AUC')
    plt.grid(True)

    # Calculate mean AUC difference
    mean_auc_diff = abs(np.mean(auc_diff)*100)
    
    # Annotate mean AUC difference on the plot   
    plt.text(1.0, 1.01, f'mean AUC diff: {mean_auc_diff:.2f} %', transform=plt.gca().transAxes, fontsize=12, ha='right', va='bottom')    
    plt.legend()
    plt.savefig(f"esport/{model_file}/cross_val_auc_plot.png")
    plt.show()
    plt.clf()





# LEGACY FUNC
def features_laso(X,y):

    alphas = np.logspace(-10, 10, 3)
    # Use LassoCV for cross-validated alpha selection with a wide range of alphas
    lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=1000000)
    lasso_cv.fit(X, y)
    # Get the best alpha value
    best_alpha_cv = lasso_cv.alpha_
    # Train the model with the best alpha
    model = Lasso(alpha=best_alpha_cv)
    model.fit(X, y)
    # Get the coefficients and corresponding feature names
    coefficients = model.coef_
    feature_names = list(X.columns)  # Assuming X is a pandas DataFrame
    # Get the indices of non-zero coefficients (selected features)
    selected_feature_indices = np.where(coefficients != 0)[0]
    # Get the names of selected features
    selected_features = [feature_names[i] for i in selected_feature_indices]

    return selected_features


def features_ref(X,y):

    rfe = RFE(model, n_features_to_select=10)
    rfe.fit(X, y)   

    features = list(rfe.support_)
    features_name = [i for i in X.columns]

    features_df = pd.DataFrame({
        'features_name': features_name,
        'selected': features
    })

    selected_features = features_df.loc[features_df['selected'], 'features_name'].tolist()
    return selected_features