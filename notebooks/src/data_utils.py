import json
from pydoc import classname
import numpy as np
import pandas as pd


def load_csv_plan(path):
    with open(path, "r") as f:
        return json.load(f)


def load_tabular_features(path):
    data = np.load(path, allow_pickle=True)

    return {
        "X": data["X"],
        "y": data["y"],
        "labels": data["labels"],
        "folds": data["folds"],
        "filenames": data["filenames"],
        "feature_names": data["feature_names"],
    }


def load_logmel_features(path):
    data = np.load(path, allow_pickle=True)

    return {
        "X": data["X"].astype(np.float32),
        "y": data["y"].astype(np.int64),
        "labels": data["labels"],
        "folds": data["folds"].astype(np.int64),
        "filenames": data["filenames"],
    }


def make_class_table(y, labels):
    return (
        pd.DataFrame({"class_id": y, "class": labels})
        .drop_duplicates()
        .sort_values("class_id")
        .reset_index(drop=True)
    )


def make_class_mappings(y, labels):
    class_table = make_class_table(y, labels)

    class_ids = class_table["class_id"].to_numpy()
    class_names = class_table["class"].to_numpy()
    id_to_class = dict(zip(class_ids, class_names))

    return class_table, class_ids, class_names, id_to_class
