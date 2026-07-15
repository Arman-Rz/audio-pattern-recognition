from pathlib import Path
from pydoc import classname

from annotated_types import MinLen
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix


def save_figure(fig, filename, output_dir, dpi=300):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    path = output_dir / filename
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


def plot_single_metric_bar(
    comparison_df, metric, model_col="model", title=None, ylabel="Score", figsize=(7, 4)
):
    if title is None:
        title = metric

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(comparison_df[model_col], comparison_df[metric])

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_ids,
    class_names,
    title="Confusion Matrix",
    normalize=False,
    figsize=(9, 8),
    annotate=True,
):
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(
            cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
        )
    else:
        cm_plot = cm

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, aspect="auto")
    fig.colorbar(im, ax=ax, label="Proportion" if normalize else "Number of clips")

    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    if annotate:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                if normalize:
                    value = cm_plot[i, j]
                    text = f"{value:.2f}" if value > 0 else ""
                else:
                    value = cm[i, j]
                    text = str(value) if value > 0 else ""

                if text:
                    ax.text(j, i, text, ha="center", va="center", fontsize=8)

    fig.tight_layout()
    return fig, ax, cm


def plot_feature_importance(
    feature_importance_df,
    feature_col="feature",
    importance_col="importance",
    top_n=20,
    title="Random Forest Feature Importance",
    figsize=(9, 7),
):
    """
    Plots top-N feature importances.
    """

    top_features = (
        feature_importance_df.sort_values(importance_col, ascending=False)
        .head(top_n)
        .iloc[::-1]
    )

    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(top_features[feature_col], top_features[importance_col])

    ax.set_title(title)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_training_history(history, title_prefix="CNN", figsize=(8, 4)):
    """
    Plots training/validation loss and accuracy from one CNN history dictionary.
    """

    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig_loss, ax_loss = plt.subplots(figsize=figsize)
    ax_loss.plot(epochs, history["train_loss"], marker="o", label="Train loss")
    ax_loss.plot(
        epochs, history["validation_loss"], marker="o", label="Validation loss"
    )
    ax_loss.set_title(f"{title_prefix} Loss Curve")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    fig_loss.tight_layout()

    fig_acc, ax_acc = plt.subplots(figsize=figsize)
    ax_acc.plot(epochs, history["train_accuracy"], marker="o", label="Train accuracy")
    ax_acc.plot(
        epochs, history["validation_accuracy"], marker="o", label="Validation accuracy"
    )

    if "validation_macro_f1" in history:
        ax_acc.plot(
            epochs,
            history["validation_macro_f1"],
            marker="o",
            label="Validation macro F1",
        )

    ax_acc.set_title(f"{title_prefix} Accuracy / F1 Curve")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Score")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    fig_acc.tight_layout()

    return (fig_loss, ax_loss), (fig_acc, ax_acc)


def plot_average_training_history(histories, title_prefix="CNN", figsize=(8, 4)):
    """
    Plots average training curves over multiple folds.
    """

    if isinstance(histories, dict):
        histories_list = list(histories.values())
    else:
        histories_list = list(histories)

    min_len = min(len(h["train_loss"]) for h in histories_list)
    epochs = np.arange(1, min_len + 1)

    def avg(key):
        return np.mean([h[key][:min_len] for h in histories_list], axis=0)

    fig_loss, ax_loss = plt.subplots(figsize=figsize)
    ax_loss.plot(epochs, avg("train_loss"), marker="o", label="Train loss")
    ax_loss.plot(epochs, avg("validation_loss"), marker="o", label="Validation loss")
    ax_loss.set_title(f"{title_prefix} Average Loss Curve")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)
    fig_loss.tight_layout()

    fig_acc, ax_acc = plt.subplots(figsize=figsize)
    ax_acc.plot(epochs, avg("train_accuracy"), marker="o", label="Train accuracy")
    ax_acc.plot(
        epochs, avg("validation_accuracy"), marker="o", label="Validation accuracy"
    )

    if "validation_macro_f1" in histories_list[0]:
        ax_acc.plot(
            epochs, avg("validation_macro_f1"), marker="o", label="Validation macro F1"
        )

    ax_acc.set_title(f"{title_prefix} Average Accuracy / F1 Curve")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Score")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)
    fig_acc.tight_layout()

    return (fig_loss, ax_loss), (fig_acc, ax_acc)


def plot_embedding(
    embedding,
    labels,
    title="2D Feature Embedding",
    xlabel="Dimension 1",
    ylabel="Dimension 2",
    figsize=(9, 7),
    point_size=8,
):
    """
    Plots a 2D embedding such as PCA, t-SNE, or UMAP.
    """

    labels = np.asarray(labels)
    unique_labels = np.unique(labels)

    fig, ax = plt.subplots(figsize=figsize)

    for label in unique_labels:
        idx = labels == label
        ax.scatter(
            embedding[idx, 0],
            embedding[idx, 1],
            s=point_size,
            alpha=0.6,
            label=str(label),
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(markerscale=2, fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(alpha=0.2)

    fig.tight_layout()
    return fig, ax


def plot_class_distribution(labels, title="Class Distribution", figsize=(10, 5)):
    """
    Plots class counts.
    """

    counts = pd.Series(labels).value_counts().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(counts.index.astype(str), counts.values)

    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of clips")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax


def plot_gradcam_example(logmel, cam, true_label, pred_label, title=None):
    if title is None:
        title = f"True: {true_label} | Predicted: {pred_label}"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(logmel, origin="lower", aspect="auto")
    axes[0].set_title("Log-Mel Spectoram")
    axes[0].set_xlabel("Time frame")
    axes[0].set_ylabel("Mel bin")

    axes[1].imshow(cam, origin="lower", aspect="auto")
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].set_xlabel("Time frame")
    axes[1].set_ylabel("Mel bin")

    axes[2].imshow(logmel, origin="lower", aspect="auto")
    axes[2].imshow(cam, origin="lower", aspect="auto", alpha=0.45)
    axes[2].set_title("Overlay")
    axes[2].set_xlabel("Time frame")
    axes[2].set_ylabel("Mel bin")

    fig.suptitle(title)
    fig.tight_layout()

    return fig, axes


def plot_metric_comparison_with_error_bars(
    comparison_df,
    model_col="model",
    metrics=None,
    title="Cross-Validation Performance Comparison",
    ylabel="Score",
    figsize=(9, 5),
    capsize=5,
):
    """
    Plots grouped bar chart comparing model metrics with standart deviation error
    """

    if metrics is None:
        metrics = [
            ("accuracy_mean", "accuracy_std", "Accuracy"),
            ("f1_macro_mean", "f1_macro_std", "Macro F1"),
            ("f1_weighted_mean", "f1_weighted_std", "Weighted F1"),
        ]

    df = comparison_df.copy()

    x = np.arange(len(df[model_col]))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=figsize)

    for i, (mean_col, std_col, label) in enumerate(metrics):
        offset = (i - (len(metrics) - 1) / 2) * width

        ax.bar(
            x + offset,
            df[mean_col],
            width,
            yerr=df[std_col],
            capsize=capsize,
            label=label,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels(df[model_col])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax
