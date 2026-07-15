"""
evaluation.py

Reusable evaluation utilities for classical machine learning models.

Contents:
    - compute_metrics()
    - run_predefined_fold_cv()
    - summarize_fold_metrics()
    - make_confusion_matrix_df()
    - make_classification_report_df()

"""

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)


def compute_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true, y_pred, average="weighted", zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def run_predefined_fold_cv(
    model_fn, X, y, folds, filenames, cv_plan, id_to_class, verbose=True
):
    fold_metrics = []
    all_predictions = []

    for item in cv_plan:
        test_fold = int(item["test_fold"])
        train_folds = [int(f) for f in item["train_folds"]]

        train_idx = np.isin(folds, train_folds)
        test_idx = folds == test_fold

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_fn()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        metrics = compute_metrics(y_test, y_pred)

        fold_metrics.append(
            {
                "test_fold": test_fold,
                "n_train": int(train_idx.sum()),
                "n_test": int(test_idx.sum()),
                **metrics,
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "filename": filenames[test_idx],
                "fold": folds[test_idx],
                "true_class_id": y_test,
                "pred_class_id": y_pred,
                "true_class": [id_to_class[i] for i in y_test],
                "pred_class": [id_to_class[i] for i in y_pred],
            }
        )

        all_predictions.append(fold_predictions)

        if verbose:
            print(
                f"Fold {test_fold}: "
                f"accuracy={metrics['accuracy']:.4f}, "
                f"macro_f1={metrics['f1_macro']:.4f}, "
                f"weighted_f1={metrics['f1_weighted']:.4f}"
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    return fold_metrics_df, predictions_df


def summarize_fold_metrics(fold_metrics_df):
    metric_columns = [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro",
        "precision_weighted",
        "recall_weighted",
        "f1_weighted",
    ]

    summary = {}

    for metric in metric_columns:
        summary[f"{metric}_mean"] = float(fold_metrics_df[metric].mean())
        summary[f"{metric}_std"] = float(fold_metrics_df[metric].std())

    return summary


def make_confusion_matrix_df(y_true, y_pred, class_ids, class_names):
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)
    return cm, pd.DataFrame(cm, index=class_names, columns=class_names)


def make_classification_report_df(y_true, y_pred, class_ids, class_names):
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )

    return pd.DataFrame(report).T
