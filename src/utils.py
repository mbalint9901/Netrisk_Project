"""
Data Science Utility Functions
============================

This module contains utility functions for data science workflows including
data loading, preprocessing, feature engineering, and model evaluation.

Author: [Your Name]
Date: November 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve
from typing import Dict, List, Tuple, Optional, Union
import logging


def clean_names(df):
    """Convert column names to snake_case"""
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace(r'[^\w]', '', regex=True)
    return df

def target_encode_log_odds(df, col, target, alpha=0.5):
    tab = pd.crosstab(df[col], df[target])
    if 0 not in tab.columns or 1 not in tab.columns:
        return df[col].value_counts(normalize=True)

    n1 = tab[1] + alpha
    n0 = tab[0] + alpha
    lor = np.log(n1 / n0)
    return lor

def apply_log_odds_encoding(train_df, test_df, col, target):
    enc = target_encode_log_odds(train_df, col, target)
    train_encoded = train_df[col].map(enc).fillna(0)
    test_encoded = test_df[col].map(enc).fillna(0)
    return train_encoded, test_encoded

def apply_one_hot_encoding(train_df, test_df, categorical_cols, max_categories=50):
    """Apply one-hot encoding to categorical variables"""
    import pandas as pd
    
    # Get all unique categories from training set
    train_features = []
    test_features = []
    feature_names = []
    
    for col in categorical_cols:
        if col not in train_df.columns:
            continue
            
        # Get top categories from training set to avoid too many features
        top_categories = train_df[col].value_counts().head(max_categories).index.tolist()
        
        # Remove "No data" levels from encoding
        top_categories = [cat for cat in top_categories if cat != "No data"]
        
        # Create one-hot encoded features
        for category in top_categories:
            feature_name = f"{col}_{category}"
            train_feature = (train_df[col] == category).astype(int)
            test_feature = (test_df[col] == category).astype(int)
            
            train_features.append(train_feature)
            test_features.append(test_feature)
            feature_names.append(feature_name)
    
    # Combine into DataFrames
    train_encoded = pd.concat(train_features, axis=1, keys=feature_names)
    test_encoded = pd.concat(test_features, axis=1, keys=feature_names)
    
    return train_encoded, test_encoded

def log_odds_per_category(df, cat_col, target_col, alpha=1e-6, min_total_count=50):
    tab_counts = pd.crosstab(df[cat_col], df[target_col])
    if not set([0, 1]).issubset(tab_counts.columns):
        return pd.DataFrame()

    total = tab_counts[0] + tab_counts[1]
    mask = total >= min_total_count
    tab_counts = tab_counts[mask]

    if tab_counts.empty:
        return pd.DataFrame()

    tab_probs = pd.crosstab(df[cat_col], df[target_col], normalize="columns").loc[tab_counts.index]
    p1 = tab_probs[1]
    p0 = tab_probs[0]
    lor = np.log((p1 + alpha) / (p0 + alpha))

    out = pd.DataFrame({
        "variable": cat_col,
        "category_value": lor.index.astype(str),
        "log_odds": lor.values,
        "count_buyer": tab_counts[1].values,
        "count_nonbuyer": tab_counts[0].values,
    })

    out["total_count"] = out["count_buyer"] + out["count_nonbuyer"]
    out["abs_log_odds"] = out["log_odds"].abs()
    out["type"] = "categorical_value"
    out = out.sort_values("abs_log_odds", ascending=False)
    return out

def numeric_log_odds(df, num_col, target_col):
    x = df[[num_col]].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return None

    y = df.loc[x.index, target_col].astype(int)
    std = x[num_col].std()
    if std == 0 or np.isnan(std):
        return None

    lr = LogisticRegression(max_iter=2000)
    lr.fit(x, y)

    beta = lr.coef_[0][0]
    standardized_log_odds = beta * std

    out = pd.DataFrame({
        "variable": [num_col],
        "category_value": [f"{num_col} (continuous)"],
        "log_odds": [standardized_log_odds],
    })
    out["abs_log_odds"] = out["log_odds"].abs()
    out["type"] = "numeric_continuous"
    return out

def unified_value_level_diff(df, target_col, numeric_cols, categorical_cols, alpha=0.5, min_total_count=50):
    results = []

    for col in categorical_cols:
        if col not in df.columns:
            continue
        tmp = log_odds_per_category(df=df, cat_col=col, target_col=target_col, alpha=alpha, min_total_count=min_total_count)
        if tmp is not None and not tmp.empty:
            results.append(tmp)

    for col in numeric_cols:
        if col not in df.columns:
            continue
        tmp = numeric_log_odds(df=df, num_col=col, target_col=target_col)
        if tmp is not None:
            results.append(tmp)

    if not results:
        return pd.DataFrame()

    all_df = pd.concat(results, ignore_index=True)
    all_df = all_df.sort_values("abs_log_odds", ascending=False)
    return all_df