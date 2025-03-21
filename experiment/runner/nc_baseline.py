import sys
sys.path.append('D:\\AI\\Graduation_Project\\model\\BATM')
import os
import ast
from pathlib import Path
from itertools import product
from experiment.config import ConfigParser
from experiment.config import init_args, customer_args, set_seed
from experiment.runner.nc_base import run, init_data_loader, topic_evaluation

# setup default values
DEFAULT_VALUES = {
    "seeds": [40, 5],
    # 这里灵活设置head的数量
    "head_num": [10, 30, 50, 70, 100, 150, 180, 200],
    "embedding_type": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "xlnet-base-cased",
                       "longformer-base-4096", "transfo-xl-wt103"]
}


# if __name__ == "__main__":
#     baseline_args = [
#         {"flags": ["-ss", "--seeds"], "type": str, "target": None},
#         {"flags": ["-aa", "--arch_attr"], "type": str, "target": None},
#         {"flags": ["-va", "--values"], "type": str, "target": None},
#         {"flags": ["-tp", "--evaluate_topic"], "type": int, "target": None},
#     ]
#     args, options = init_args(), customer_args(baseline_args)
#     config_parser = ConfigParser.from_args(args, options)
#     config = config_parser.config
#     saved_dir = Path(config.project_root) / "saved" / "performance"  # init saved directory
#     os.makedirs(saved_dir, exist_ok=True)  # create empty directory
#     arch_attr = config.get("arch_attr", "base")  # test an architecture attribute
#     saved_name = f'{config.data_config["name"].replace("/", "_")}_{arch_attr}'
#     evaluate_topic, entropy_constraint = config.get("evaluate_topic", 0), config.get("entropy_constraint", 0)
#     if evaluate_topic:
#         saved_name += "_evaluate_topic"
#     if entropy_constraint:
#         saved_name += "_entropy_constraint"
#     # acquires test values for a given arch attribute
#     test_values = config.get("values").split(",") if hasattr(config, "values") else DEFAULT_VALUES.get(arch_attr, [0])
#     seeds = [int(s) for s in config.seeds.split(",")] if hasattr(config, "seeds") else DEFAULT_VALUES.get("seeds")
#     for value, seed in product(test_values, seeds):
#         try:
#             # 这个函数非常有用，因为它可以用来处理那些不包含任何可执行代码的字符串，但需要以一种安全的方式进行。它主要用于处理数字、字符串、元组、列表、字典、集合和 None。
#             config.set(arch_attr, ast.literal_eval(value))  # convert to int or float if it is a numerical value
#         except ValueError:
#             config.set(arch_attr, value)
#         config.set("seed", seed)
#         log = {"arch_type": config.arch_config["type"], "seed": config.seed, arch_attr: value,
#                "variant_name": config.arch_config.get("variant_name", None)}
#         set_seed(log["seed"])
#         data_loader = init_data_loader(config_parser)
#         trainer = run(config_parser, data_loader)
#         # log.update(test(trainer, data_loader))
#         log.update(trainer.test(trainer.best_model, data_loader))
#         if evaluate_topic:
#             topic_path = Path(config.project_root) / "saved" / "topics" / saved_name / f"{value}_{seed}"
#             log.update(topic_evaluation(trainer, data_loader, topic_path))
#         trainer.save_log(log, saved_path=saved_dir / f'{saved_name}.csv')

if __name__ == "__main__":
    head_num = DEFAULT_VALUES.get("head_num")
    baseline_args = [
        {"flags": ["-ss", "--seeds"], "type": str, "target": None},
        {"flags": ["-aa", "--arch_attr"], "type": str, "target": None},
        {"flags": ["-va", "--values"], "type": str, "target": None},
        {"flags": ["-tp", "--evaluate_topic"], "type": int, "target": None},
    ]
    args, options = init_args(), customer_args(baseline_args)
    config_parser = ConfigParser.from_args(args, options)
    config = config_parser.config
    saved_dir = Path(config.project_root) / "saved" / "performance"  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", "base")  # test an architecture attribute
    saved_name = f'{config.data_config["name"].replace("/", "_")}_{arch_attr}'
    evaluate_topic, entropy_constraint = config.get("evaluate_topic", 0), config.get("entropy_constraint", 0)
    if evaluate_topic:
        saved_name += "_evaluate_topic"
    if entropy_constraint:
        saved_name += "_entropy_constraint"
    # acquires test values for a given arch attribute
    test_values = config.get("values").split(",") if hasattr(config, "values") else DEFAULT_VALUES.get(arch_attr, [0])
    seeds = [int(s) for s in config.seeds.split(",")] if hasattr(config, "seeds") else DEFAULT_VALUES.get("seeds")
    for value, seed in product(test_values, seeds):
        try:
            # 这个函数非常有用，因为它可以用来处理那些不包含任何可执行代码的字符串，但需要以一种安全的方式进行。它主要用于处理数字、字符串、元组、列表、字典、集合和 None。
            config.set(arch_attr, ast.literal_eval(value))  # convert to int or float if it is a numerical value
        except ValueError:
            config.set(arch_attr, value)
        config.set("seed", seed)
        log = {"arch_type": config.arch_config["type"], "seed": config.seed, arch_attr: value,
               "variant_name": config.arch_config.get("variant_name", None)}
        set_seed(log["seed"])
        data_loader = init_data_loader(config_parser)
        trainer = run(config_parser, data_loader)
        # log.update(test(trainer, data_loader))
        log.update(trainer.test(trainer.best_model, data_loader))
        if evaluate_topic:
            topic_path = Path(config.project_root) / "saved" / "topics" / saved_name / f"{value}_{seed}"
            log.update(topic_evaluation(trainer, data_loader, topic_path))
        trainer.save_log(log, saved_path=saved_dir / f'{saved_name}.csv')
