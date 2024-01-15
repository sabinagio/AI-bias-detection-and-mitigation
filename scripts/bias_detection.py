import pandas as pd
import json 
from preprocess import *

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import roc_auc_score


## ROC-AUC diagnostic

def clf(X, y, text_col):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[text_col])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[text_col])

    # Create the RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train_tfidf, y_train)

    # Make predictions on the test set
    y_pred = rf_classifier.predict(X_test_tfidf)
    y_pred_proba = rf_classifier.predict_proba(X_test_tfidf)

    # Evaluate the classifier
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report for detailed metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, accuracy


def calculate_auc(y_true, y_pred, class_labels, multi_class=True, subgroup_mask=None, background_mask=None):
    """
    Calculate AUC for different subsets.

    Parameters:
    - y_true: True labels (binary).
    - y_pred: Predicted probabilities or scores.
    - subgroup_mask: Mask for the subgroup examples.
    - background_mask: Mask for the background examples.

    Returns:
    - Overall AUC and Bias AUCs (Subgroup AUC, BPSN AUC, BNSP AUC).
    """

    # Overall AUC
    if multi_class == False:
        overall_auc = roc_auc_score(y_true, y_pred)
    else:
        overall_auc = roc_auc_score(y_true, y_pred, multi_class='ovo', labels=class_labels)

    # Subgroup AUC
    if subgroup_mask is not None:
        if multi_class == False:
            subgroup_auc = roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask])
        else:
            subgroup_auc = roc_auc_score(y_true[subgroup_mask], y_pred[subgroup_mask], multi_class='ovo', labels=class_labels)
    else:
        subgroup_auc = np.nan

    # BPSN AUC
    if background_mask is not None and subgroup_mask is not None:
        if multi_class == False:
            bpsn_mask = np.logical_and(background_mask, ~subgroup_mask)
            bpsn_auc = roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask])
        else:
            bpsn_mask = np.logical_and(background_mask, ~subgroup_mask)
            bpsn_auc = roc_auc_score(y_true[bpsn_mask], y_pred[bpsn_mask], multi_class='ovo', labels=class_labels)
    else:
        bpsn_auc = np.nan

    # BNSP
    if background_mask is not None and subgroup_mask is not None:
        if multi_class == False:
            bnsp_mask = np.logical_and(~background_mask, subgroup_mask)
            bnsp_auc = roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask])
        else:
            bnsp_mask = np.logical_and(~background_mask, subgroup_mask)
            bnsp_auc = roc_auc_score(y_true[bnsp_mask], y_pred[bnsp_mask], multi_class='ovo', labels=class_labels)
    else:
        bnsp_auc = np.nan

    return overall_auc, subgroup_auc, bpsn_auc, bnsp_auc


def equal_accuracy_multiclass(df, text_col, subgroup):

    '''
    subgroups should be a list even if there is only one value 
    '''

    # Background group
    X_privileged = df[df[subgroup]==0][[text_col, subgroup]]
    y_privileged = df[df[subgroup]==0].label

    labels = sorted(list(df.label.unique()))

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, privileged_group_acc = clf(X_privileged, y_privileged, text_col)
    print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))

    # Subgroup
    X_unprivileged = df[df[subgroup]==1][[text_col, subgroup]]
    y_unprivileged = df[df[subgroup]==1].label

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, subgroup_accuracy = clf(X_unprivileged, y_unprivileged, text_col)
    print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))

    acc_delta = subgroup_accuracy - privileged_group_acc

    return privileged_group_acc, subgroup_accuracy, acc_delta


def equal_accuracy_binary(df, text_col, subgroup):
    '''
    subgroups should be a list even if there is only one value 
    '''

    # Background group
    X_privileged = df[df[subgroup]==0][[text_col, subgroup]]
    y_privileged = df[df[subgroup]==0].label

    labels = sorted(list(df.label.unique()))

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, privileged_group_acc = clf(X_privileged, y_privileged, text_col)
    print(roc_auc_score(y_test, y_pred_proba, labels=labels))

    # Subgroup
    X_unprivileged = df[df[subgroup]==1][[text_col, subgroup]]
    y_unprivileged = df[df[subgroup]==1].label

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, subgroup_accuracy = clf(X_unprivileged, y_unprivileged, text_col)
    print(roc_auc_score(y_test, y_pred_proba, labels=labels))

    acc_delta = subgroup_accuracy - privileged_group_acc

    return privileged_group_acc, subgroup_accuracy, acc_delta

def detect_bias_subgroup(df, text_col, subgroup, multi_class=True):

    df = df.dropna(subset=[text_col])
    df = df.dropna(subset=[subgroup])

    X = df[[text_col, subgroup]]    
    y = df.label # has to change and be arg in the future
    labels = sorted(list(df.label.unique()))

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, accuracy = clf(X, y, text_col)

    overall_auc = accuracy
    print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))

    subgroup_mask = X_test[subgroup] == 1
    background_group_mask = X_test[subgroup] == 0

    overall_auc, subgroup_auc, bpsn_auc, bnsp_auc = calculate_auc(
        y_test, 
        y_pred_proba, 
        labels, 
        multi_class=True, 
        subgroup_mask=subgroup_mask, 
        background_mask=background_group_mask
    )

    print("Overall AUC:", overall_auc)
    print("Subgroup AUC:", subgroup_auc)
    ''' 
    Subgroup AUC: Restrict the test set to examples that mention this subgroup. 
    A low value means that the model does poorly at distinguishing hate-speech and non-hate-speech examples 
    that mention where this subgroup is present.

    '''
    print("BPSN AUC:", bpsn_auc)
    ''' 
    BPSN (Background Positive, Subgroup Negative) AUC: 
    Restrict the test set to non-hate-specch examples that mention this subgroup and hate-speech examples that do not. 
    A low value suggests that the models scores skew higher than they should for examples mentioning this subgroup.
    ''' 
    print("BNSP AUC:", bnsp_auc)
    ''' 
    BNSP (Background Negative, Subgroup Positive) AUC:
    Restrict the test set to hate-speech examples that mention this subgroup and non-hate-speech examples that do not. 
    A low value suggests that the models scores skew lower than they should for examples mentioning this subgroup.
    ''' 

    if multi_class==True:
        privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_multiclass(df, text_col, subgroup)
    else:
        privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_binary(df, text_col, subgroup)

    return overall_auc, subgroup_auc, bpsn_auc, bnsp_auc, privileged_group_acc, subgroup_accuracy, acc_delta


def detect_bias_all_subgroups(df, text_col, subgroups, multi_class=True):
    df = df.dropna(subset=[text_col])
    
    bias_dict = {}
    subgroup_accuracies = []
    privileged_group_acc = 0
    for subgroup in subgroups:
        overall_auc, subgroup_auc, bpsn_auc, bnsp_auc, privileged_group_acc, subgroup_accuracy, acc_delta = detect_bias_subgroup(df, text_col, subgroup, multi_class=True)
        subgroup_accuracies.append(subgroup_accuracy)
        bias_dict['subgroup'] = {
            'overall_auc': overall_auc, 
            'subgroup_auc': subgroup_auc, 
            'bpsn_auc': bpsn_auc, 
            'bnsp_auc': bnsp_auc, 
            'privileged_group_acc': privileged_group_acc, 
            'subgroup_accuracy': subgroup_accuracies, 
            'acc_delta': acc_delta
        }
        privileged_group_acc += privileged_group_acc
    mean_delta = mean(subgroup_accuracies) - privileged_group_acc