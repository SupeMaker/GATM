import sys
sys.path.append('D:\\AI\\Graduation_Project\\model\\BATM')
# sys.path.append('/kaggle/input/batm-code5/BATM')
import os
import numpy as np
import models as module_arch
import experiment.data_loader as module_data
from typing import Union
from torch.backends import cudnn
from scipy.stats import entropy
from experiment.data_loader import NewsDataLoader
from experiment.trainer import NCTrainer
from experiment.config import ConfigParser, init_args, customer_args, set_seed
from utils.topic_utils import get_topic_dist, save_topic_info


def init_default_model(config_parser: ConfigParser, data_loader: NewsDataLoader):
    # build a default model architecture
    # word_dict获得词表中每个词的索引列表
    model_params = {"num_classes": len(data_loader.label_dict), "word_dict": data_loader.word_dict}
    # 如果 object 对象具有名为 name 的属性或方法，则 hasattr() 函数返回 True。如果 object 对象不具有名为 name 的属性或方法，则返回 False
    # 查看 NewsDataLoader类， 如果存在 embeds这个属性，则说明embedding_type == "glove"
    if hasattr(data_loader, "embeds"):
        # 可以看 NewsDataLoader的代码，如果有这个属性说明为glove嵌入
        model_params.update({"embeds": data_loader.embeds})
        # arch_config的定义在 /home/zhouyonglin/work/model/BATM/experiment/config/configuration.py
        # 对应默认返回的模型为baseline
    model = config_parser.init_obj("arch_config", module_arch, **model_params)
    return model


def init_data_loader(config_parser: ConfigParser):
    # setup data_loader instances
    # 从 data_config(/home/zhouyonglin/work/model/BATM/experiment/config/configuration.py)中取出type,这里的type是指类NewsDataLoader，返回的就是这个类
    data_loader = config_parser.init_obj("data_config", module_data)
    return data_loader


def run(config_parser: ConfigParser, data_loader: NewsDataLoader):
    cudnn.benchmark = False
    cudnn.deterministic = True
    logger = config_parser.get_logger("train")
    model = init_default_model(config_parser, data_loader)
    logger.info(model)
    trainer = NCTrainer(model, config_parser, data_loader)
    # 训练模型
    trainer.train()
    return trainer


# def test(trainer: NCTrainer, data_loader: NewsDataLoader):
#     log = {}
#     # run validation
#     log.update(trainer.evaluate(data_loader.valid_loader, trainer.best_model, prefix="val"))
#     # run test
#     log.update(trainer.evaluate(data_loader.test_loader, trainer.best_model, prefix="test"))
#     return log


def topic_evaluation(trainer: NCTrainer, data_loader: NewsDataLoader, path: Union[str, os.PathLike]):
    # statistic topic distribution of Topic Attention network
    reverse_dict = {v: k for k, v in data_loader.word_dict.items()}
    topic_dist = get_topic_dist(trainer, list(data_loader.word_dict.values()))
    topic_result = save_topic_info(path, topic_dist, reverse_dict, data_loader)
    topic_result.update({"token_entropy": np.mean(entropy(topic_dist, axis=1))})
    return topic_result


if __name__ == "__main__":
    args, options = init_args(), customer_args()
    main_config = ConfigParser.from_args(args, options)
    set_seed(main_config["seed"])
    trainer = run(main_config, init_data_loader(main_config))
