"""
cnn.py

Utility functions and classes for CNN-based audio classification.

Contents:
    - LogMelDataset
    - AudioCNN
    - train_one_epoch()
    - ecaluate()
    - train_cnn_cv()
"""

import pandas as pd
import numpy as np
import torch
import copy
import random
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

DEFAULT_CNN_CONFIG = {
    "seed": 42,
    "batch_size": 64,
    "max_epoch": 40,
    "patience": 7,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "dropout": 0.5,
    "validation_size": 0.2,
    "num_workers": 0,
}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LogMelDataset(Dataset):
    def __init__(self, X, y, mean, std):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = (self.X[idx] - self.mean) / self.std
        x = np.expand_dims(x, axis=0)
        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(self.y[idx], dtype=torch.long),
        )


class AudioCNN(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)

        y_true.extend(yb.detach().cpu().numpy())
        y_pred.extend(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)

    return avg_loss, acc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true, y_pred = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(yb.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(y_true, y_pred)
    return (
        avg_loss,
        acc,
        np.array(y_true),
        np.array(y_pred),
    )


def train_cnn_cv(
    X,
    y,
    folds,
    filenames,
    cv_plan,
    id_to_class,
    num_classes,
    device,
    config=None,
    verbose=True,
    test_folds=None,
):
    config = {**DEFAULT_CNN_CONFIG, **(config or {})}
    set_seed(config["seed"])

    fold_metrics = []
    all_predictions = []
    all_histories = {}

    best_overall_f1 = -1.0
    best_overall_state = None
    best_overall_fold = None

    for item in cv_plan:
        test_fold = int(item["test_fold"])

        if test_folds is not None and test_fold not in test_folds:
            continue

        train_folds = [int(f) for f in item["train_folds"]]

        print("\n" + "=" * 60)
        print(f"Fold {test_fold}")
        print("=" * 60)

        train_pool_idx = np.where(np.isin(folds, train_folds))[0]
        test_idx = np.where(folds == test_fold)[0]

        train_idx, val_idx = train_test_split(
            train_pool_idx,
            test_size=config["validation_size"],
            random_state=config["seed"],
            stratify=y[train_pool_idx],
        )

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        train_mean = X_train.mean()
        train_std = X_train.std() + 1e-8

        train_ds = LogMelDataset(X_train, y_train, train_mean, train_std)
        val_ds = LogMelDataset(X_val, y_val, train_mean, train_std)
        test_ds = LogMelDataset(X_test, y_test, train_mean, train_std)

        train_loader = DataLoader(
            train_ds,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=torch.cuda.is_available(),
        )

        model = AudioCNN(num_classes, dropout=config["dropout"]).to(device)

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
        )

        best_fold_state = None
        best_fold_f1 = -1.0
        patience_counter = 0

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
            "validation_macro_f1": [],
        }

        for epoch in range(1, config["max_epoch"] + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, y_val_true, y_val_pred = evaluate(
                model, val_loader, criterion, device
            )

            _, _, val_macro_f1, _ = precision_recall_fscore_support(
                y_val_true, y_val_pred, average="macro", zero_division=0
            )

            history["train_loss"].append(float(train_loss))
            history["train_accuracy"].append(float(train_acc))
            history["validation_loss"].append(float(val_loss))
            history["validation_accuracy"].append(float(val_acc))
            history["validation_macro_f1"].append(float(val_macro_f1))

            if verbose:
                print(
                    f"Epoch {epoch:02d}: "
                    f"train_loss={train_loss:.4f}, "
                    f"train_acc={train_acc:.4f}, "
                    f"val_loss={val_loss:.4f}, "
                    f"val_accuracy={val_acc:.4f}, "
                    f"val_macro_f1={val_macro_f1:.4f}"
                )

            if val_macro_f1 > best_fold_f1:
                best_fold_f1 = val_macro_f1
                best_fold_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= config["patience"]:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

        model.load_state_dict(best_fold_state)
        test_loss, test_acc, y_true, y_pred = evaluate(
            model, test_loader, criterion, device
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )

        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        fold_metrics.append(
            {
                "test_fold": test_fold,
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "n_test": int(test_idx.sum()),
                "accuracy": float(test_acc),
                "precision_macro": float(precision_macro),
                "recall_macro": float(recall_macro),
                "f1_macro": float(f1_macro),
                "precision_weighted": float(precision_weighted),
                "recall_weighted": float(recall_weighted),
                "f1_weighted": float(f1_weighted),
                "best_epoch": int(np.argmax(history["validation_macro_f1"]) + 1),
                "best_validation_macro_f1": float(best_fold_f1),
            }
        )

        fold_predictions = pd.DataFrame(
            {
                "filename": filenames[test_idx],
                "fold": folds[test_idx],
                "true_class_id": y_true,
                "pred_class_id": y_pred,
                "true_class": [id_to_class[i] for i in y_true],
                "pred_class": [id_to_class[i] for i in y_pred],
            }
        )

        all_predictions.append(fold_predictions)
        all_histories[str(test_fold)] = history

        if f1_macro > best_overall_f1:
            best_overall_f1 = f1_macro
            best_overall_state = copy.deepcopy(model.state_dict())
            best_overall_fold = test_fold

        if verbose:
            print(
                f"Fold {test_fold} final: "
                f"accuracy={test_acc:.4f}, "
                f"macro_f1={f1_macro:.4f}"
            )

    fold_metrics_df = pd.DataFrame(fold_metrics)
    predictions_df = pd.concat(all_predictions, ignore_index=True)

    best_info = {
        "best_overall_fold": int(best_overall_fold),
        "best_overall_macro_f1": float(best_overall_fold),
    }

    return (
        fold_metrics_df,
        predictions_df,
        all_histories,
        best_overall_state,
        best_info,
    )


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(self.save_activation)
        self.backward_handle = target_layer.register_full_backward_hook(
            self.save_gradient
        )

    def save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits = self.model(x)

        if class_idx is None:
            class_idx = int(torch.argmax(logits, dim=1).item())

        score = logits[:, class_idx]
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam, class_idx, logits.detach().cpu().numpy()

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()


def prepare_logmel_tensor(logmel, mean, std, device):
    x = (logmel - mean) / std
    x = np.expand_dims(x, axis=(0, 1))

    return torch.tensor(x, dtype=torch.float32).to(device)
