import argparse
import collections
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_args():
    args = argparse.ArgumentParser(description="Define Argument")
    args.add_argument("-r", "--resume", default=None, type=str,
                      help="path to latest checkpoint (default: None)")
    # 默认 ArgumentParser(prog='nc_base.py', usage=None, description='Define Argument', formatter_class=<class 'argparse.HelpFormatter'>, conflict_handler='error', add_help=True)
    return args


def customer_args(args=None):
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple("CustomArgs", "flags type target")
    # set default arguments
    options = [
        # global variables
        CustomArgs(["-ng", "--n_gpu"], type=int, target=None),
        CustomArgs(["-s", "--seed"], type=int, target=None),
        CustomArgs(["-ml", "--max_length"], type=int, target=None),
        CustomArgs(["-sm", "--save_model"], type=int, target=None),
        CustomArgs(["-et", "--embedding_type"], type=str, target=None),
        CustomArgs(["-ext", "--extract_type"], type=str, target=None),
        CustomArgs(["-sd", "--save_dir"], type=str, target=None),
        CustomArgs(["-rn", "--run_name"], type=str, target=None), # run_name是在base_trainer.py中log["run_id"] = self.config["run_name"]设置 run_id的， 例如：MIND15/keep_all/PretrainedBaseline
        # architecture params
        CustomArgs(["-at", "--arch_type"], type=str, target="arch_config"),
        CustomArgs(["-p", "--pooling"], type=str, target="arch_config"), # 管理池化层
        CustomArgs(["-em", "--entropy_method"], type=str, target="arch_config"),
        CustomArgs(["-vn", "--variant_name"], type=str, target="arch_config"),
        CustomArgs(["-an", "--act_name"], type=str, target="arch_config"), # D:\AI\Graduation_Project\model\BATM\models\layers.py 中管理激活函数的
        CustomArgs(["-up", "--use_pretrained"], type=int, target="arch_config"),
        CustomArgs(["-nl", "--n_layers"], type=int, target="arch_config"),
        CustomArgs(["-ed", "--embedding_dim"], type=int, target="arch_config"),
        CustomArgs(["-ec", "--entropy_constraint"], type=int, target="arch_config"),
        CustomArgs(["-ce", "--calculate_entropy"], type=int, target="arch_config"),
        CustomArgs(["-ap", "--add_pos"], type=int, target="arch_config"),
        CustomArgs(["-hn", "--head_num"], type=int, target="arch_config"),
        CustomArgs(["-hd", "--head_dim"], type=int, target="arch_config"),
        CustomArgs(["-al", "--alpha"], type=float, target="arch_config"),
        CustomArgs(["-dr", "--dropout_rate"], type=float, target="arch_config"),
        # dataloader params
        CustomArgs(["-na", "--name"], type=str, target="data_config"), # 这里修改 MIND15/keep_all的地方 --name=MIND15/keep_all
        CustomArgs(["-eb", "--embed_method"], type=str, target="data_config"),
        CustomArgs(["-bs", "--batch_size"], type=int, target="data_config"),
        # trainer params
        CustomArgs(["-ep", "--epochs"], type=int, target="trainer_config"),
        # optimizer
        CustomArgs(["-lr", "--lr"], type=float, target="optimizer_config"),
        CustomArgs(["-wd", "--weight_decay"], type=float, target="optimizer_config"),
        CustomArgs(["-wg", "--with_gru"], type=str, target="arch_config"),
    ]
    if args:
        options.extend([CustomArgs(**ca) for ca in args])
    return options
