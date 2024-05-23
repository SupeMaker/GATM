import os
import pandas as pd
import numpy as np

from pathlib import Path
from itertools import product
from utils import get_project_root, del_index_column


def get_mean(values):
    return f"{np.round(np.mean(values), 2)}"+u"\u00B1"+f"{np.round(np.std(values), 2)}"


def perform_aggregate(group_by: list, **kwargs):
    stat_df = pd.DataFrame()
    root_path = Path(get_project_root()) / "saved"
    file_name = f"{kwargs.get('name')}_{kwargs.get('set_type')}_{kwargs.get('arg')}.csv"
    df_file = root_path / "performance" / file_name
    saved_path = root_path / "stat"
    os.makedirs(saved_path, exist_ok=True)
    if os.path.exists(df_file):
        per_df = del_index_column(pd.read_csv(df_file))
        for _, group in per_df.groupby(group_by):
            metrics = [f"{d}_{m}" for d, m in product(["val", "test"], ["loss", "accuracy", "macro_f"])]
            mean_values = [get_mean(group[m].values * 100) for m in metrics]
            group = group.drop(columns=metrics + ["seed", "run_id", "dropout_rate"]).drop_duplicates()
            group[metrics] = pd.DataFrame([mean_values], index=group.index)
            stat_df = stat_df.append(group, ignore_index=True)
        stat_df.to_csv(saved_path / file_name)
    return stat_df


if __name__ == "__main__":
    names = ["News26", "MIND15"]
    datasets = ["keep_all", "aggressive", "alphabet_only"]
    test_args = ["head_num", "embedding_type", "base", "variant_name"]
    for name, set_type, arg in product(names, datasets, test_args):
        args = {"name": name, "set_type": set_type, "arg": arg}
        if arg == "embedding_type":
            perform_aggregate(["arch_type", arg], **args)
        else:
            perform_aggregate(["arch_type", arg, "variant_name", "max_length"], **args)
