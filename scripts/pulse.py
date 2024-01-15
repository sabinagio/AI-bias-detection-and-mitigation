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
from scipy.sparse import hstack

from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate
from tensorflow.keras.optimizers import Adam


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
    # print(f"Accuracy: {accuracy:.4f}")

    # Print classification report for detailed metrics
    # print("Classification Report:")
    # print(classification_report(y_test, y_pred))

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
    # print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))

    # Subgroup
    X_unprivileged = df[df[subgroup]==1][[text_col, subgroup]]
    y_unprivileged = df[df[subgroup]==1].label

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, subgroup_accuracy = clf(X_unprivileged, y_unprivileged, text_col)
    # print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))
    print(subgroup_accuracy)
    print(classification_report(y_test, y_pred))

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
    # print(roc_auc_score(y_test, y_pred_proba, labels=labels))

    # Subgroup
    X_unprivileged = df[df[subgroup]==1][[text_col, subgroup]]
    y_unprivileged = df[df[subgroup]==1].label

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, subgroup_accuracy = clf(X_unprivileged, y_unprivileged, text_col)
    # print(roc_auc_score(y_test, y_pred_proba, labels=labels))
    print(subgroup_accuracy)
    print(classification_report(y_test, y_pred))

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
    # print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))
    print(classification_report(y_test, y_pred))

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

        return bias_dict, privileged_group_acc, subgroup_accuracies, mean_delta 



''' 
Bias mitigation
''' 

def clf_weights(X, y, text_col, subgroup):
    # Split the data into training, testing and subgroups sets
    X_train, X_test, y_train, y_test, sub_train, sub_test = train_test_split(
        X, y, subgroup, test_size=0.2, random_state=42
    )

    # assign 'higher' weights to the subgroup
    weights_train = sub_train.replace({0: 0, 1: 2})

    # Create a TF-IDF vectorizer
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[text_col])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[text_col])

    # Create the RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier
    rf_classifier.fit(X_train_tfidf, y_train, weights_train)

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


def correct_bias_subgroup_weights(df, text_col, subgroup, multi_class=True):

    df = df.dropna(subset=[text_col])
    df = df.dropna(subset=[subgroup])

    X = df[[text_col, subgroup]]    
    y = df.label # has to change and be arg in the future
    labels = sorted(list(df.label.unique()))

    X_train, X_test, X_train_tfidf, X_test_tfidf, y_train, y_test, rf_classifier, y_pred, y_pred_proba, accuracy = clf_weights(X, y, text_col, df[subgroup])

    overall_acc = accuracy
    # print(roc_auc_score(y_test, y_pred_proba, multi_class='ovo', labels=labels))
    # print(classification_report(y_test, y_pred))

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

    # if multi_class==True:
    #     privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_multiclass(df, text_col, subgroup)
    # else:
    #     privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_binary(df, text_col, subgroup)

    return overall_auc, subgroup_auc, bpsn_auc, bnsp_auc


# Adversarial debiasing

'''
This process aims to reduce bias in the model predictions by introducing an adversarial model that tries to predict 
the protected subgroup membership. The debiased model is then trained using the combined input, which includes the original
 features and the trimmed output of the adversary. The hope is that this adversarial training process helps the model 
 to make predictions that are less influenced by the protected subgroup. 
'''

def create_adversarial_model(input_shape):
    input_layer = Input(shape=(input_shape,), name='input_features')
    output_layer = Dense(1, activation='sigmoid', name='adversary_output')(input_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='adversary_model')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def create_debiased_model(input_shape, num_classes):
    input_layer = Input(shape=(input_shape,), name='input_features')
    output_layer = Dense(num_classes, activation='softmax', name='main_output')(input_layer)
    
    model = Model(inputs=input_layer, outputs=output_layer, name='debiased_model')
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def adversarial_debiasing(X_train_tfidf, subgroup_train_binary, y_train, X_test_tfidf, num_classes):
    input_shape = X_train_tfidf.shape[1]

    # Create and train adversarial model
    adversary_model = create_adversarial_model(input_shape)

    # Convert sparse matrices to dense arrays
    X_train_dense = X_train_tfidf.toarray()
    X_test_dense = X_test_tfidf.toarray()

    # Use 'data' instead of 'values'
    row_indices, col_indices = X_train_tfidf.nonzero()

    # Reorder the sparse matrix
    reordered_indices = tf.sparse.reorder(tf.sparse.SparseTensor(
        indices=np.column_stack([row_indices, col_indices]),
        values=X_train_tfidf.data,
        dense_shape=X_train_tfidf.shape
    ))

    adversary_model.fit(reordered_indices, subgroup_train_binary, epochs=10, verbose=0)

    # Freeze the weights of the adversary model
    adversary_model.trainable = False

    # Trim the adversary predictions to match the TF-IDF shape
    adversary_predictions_train = adversary_model.predict(X_train_dense)[:, :input_shape]

    # Combine the input features and the trimmed output of the adversary model for training
    combined_input_train = hstack([X_train_tfidf, adversary_predictions_train])

    # Create debiased model
    debiased_model = create_debiased_model(combined_input_train.shape[1], num_classes)

    # Train the debiased model with the combined input for training
    debiased_model.fit(
        combined_input_train.toarray(),
        tf.keras.utils.to_categorical(y_train, num_classes=num_classes),
        epochs=10,
        verbose=0
    )

    # Trim the adversary predictions on the test set
    adversary_predictions_test = adversary_model.predict(X_test_dense)[:, :input_shape]

    # Combine the input features and the trimmed output of the adversary model for testing
    combined_input_test = hstack([X_test_tfidf, adversary_predictions_test])

    # Make predictions on the test set
    y_pred_test = debiased_model.predict(combined_input_test.toarray())
    y_pred_proba_test = y_pred_test.flatten()

    return y_pred_test, y_pred_proba_test

def correct_bias_subgroup_ad(df, text_col, subgroup, multi_class=True):

    df = df.dropna(subset=[text_col])
    df = df.dropna(subset=[subgroup])

    X = df[[text_col, subgroup]]    
    y = df.label # has to change and be arg in the future
    labels = sorted(list(df.label.unique()))

    X_train, X_test, y_train, y_test, subgroup_train, subgroup_test = train_test_split(
        X, y, df[subgroup], test_size=0.2, random_state=42
    )

    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train[text_col])
    X_test_tfidf = tfidf_vectorizer.transform(X_test[text_col])

    le = LabelEncoder()
    subgroup_train_binary = le.fit_transform(subgroup_train)

    # Perform adversarial debiasing
    y_pred, y_pred_proba = adversarial_debiasing(X_train_tfidf, subgroup_train_binary, y_train, X_test_tfidf, len(labels))

    print(len(y_test))
    print(len(y_pred))
    print(len(y_pred_proba))

    # print(y_test)
    # print(y_pred)

    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Evaluate the predictions
    accuracy = accuracy_score(y_test, y_pred_labels)
    print(f"Accuracy: {accuracy:.4f}")

    # Print classification report for detailed metrics
    print("Classification Report:")
    print(classification_report(y_test, y_pred_labels))

    print(y_pred_proba)

    y_pred_proba_2d = y_pred_proba.reshape(-1, len(labels))

    # Evaluate bias using ROC-AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba_2d, multi_class='ovo')
    print("ROC-AUC:", roc_auc)

    subgroup_mask = X_test[subgroup] == 1
    background_group_mask = X_test[subgroup] == 0

    overall_auc, subgroup_auc, bpsn_auc, bnsp_auc = calculate_auc(
        y_test, 
        y_pred_proba_2d, 
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

    # if multi_class==True:
    #     privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_multiclass(df, text_col, subgroup)
    # else:
    #     privileged_group_acc, subgroup_accuracy, acc_delta = equal_accuracy_binary(df, text_col, subgroup)

    return overall_auc, subgroup_auc, bpsn_auc, bnsp_auc
