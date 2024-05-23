import numpy as np
import torch
from keybert import KeyBERT

from base.base_trainer import BaseTrainer
# from experiment.runner.nc_base import test
from experiment import NewsDataLoader
from utils import MetricTracker
from tqdm import tqdm


class NCTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, config, data_loader, **kwargs):
        super().__init__(model, config)
        self.config = config
        self.data_loader = data_loader.train_loader
        # Configuration类实现了 get() 函数,这里是获取 arch_config 属性
        arch_config = self.config["arch_config"]
        self.entropy_constraint = arch_config.get("entropy_constraint", False)
        self.calculate_entropy = arch_config.get("calculate_entropy", self.entropy_constraint)
        # 训练步长
        self.alpha = arch_config.get("alpha", 0.001)
        self.embedding_type = arch_config.get("embedding_type", "bert-base-uncased")
        self.len_epoch = len(self.data_loader)
        self.valid_loader = data_loader.valid_loader
        self.test_loader = data_loader.test_loader
        self.d_loader = data_loader
        self.do_validation = self.valid_loader is not None
        self.log_step = int(np.sqrt(self.data_loader.batch_size))
        metrics = ["loss"] + [m.__name__ for m in self.metric_ftns]
        if self.calculate_entropy:
            metrics.extend(["doc_entropy"])
        self.train_metrics = MetricTracker(*metrics, writer=self.writer)
        self.valid_metrics = MetricTracker(*metrics, writer=self.writer)
        # self.extract_model = KeyBERT(model='D:\\AI\\model\\' + self.embedding_type)

    # 将数据放入GPU中
    def load_batch_data(self, batch_dict):
        """
        load batch data to default device
        """
        return {k: v.to(self.device) for k, v in batch_dict.items()}

# 训练模型并返回真实结果和loss
    def run_model(self,  batch_dict, model=None):
        """
        run model with the batch data
        :param batch_dict: the dictionary of data with format like {"data": Tensor(), "label": Tensor()}
        :param model: by default we use the self model
        :return: the output of running, label used for evaluation, and loss item
        """
        # doc_embedding, word_embedding = self.extract_model.extract_embeddings(batch_dict['text'])
        # new_batch = {}
        # new_batch["data"] = batch_dict["data"]
        # new_batch["label"] = batch_dict["label"]
        # new_batch["mask"] = batch_dict["mask"]
        # new_batch["doc_embedding"] = torch.tensor(doc_embedding)
        # new_batch["word_embedding"] = torch.tensor(word_embedding)
        # batch_dict = new_batch
        # 将数据放入GPU中
        batch_dict = self.load_batch_data(batch_dict)
        # 训练模型
        output = model(batch_dict) if model is not None else self.model(batch_dict)
        loss = self.criterion(output[0], batch_dict["label"])
        out_dict = {"label": batch_dict["label"], "loss": loss, "predict": output[0]}
        # 使用商约束
        if self.entropy_constraint:
            loss += self.alpha * output[2]
        if self.calculate_entropy:
            out_dict.update({"attention_weight": output[1], "entropy": output[2]})
        return out_dict

    # 更新评价函数  ["accuracy", "macro_f"]
    def update_metrics(self, metrics, out_dict):
        n = len(out_dict["label"])
        metrics.update("loss", out_dict["loss"].item(), n=n)  # update metrix
        if self.calculate_entropy:
            metrics.update("doc_entropy", out_dict["entropy"].item() / n, n=n)
        for met in self.metric_ftns:  # run metric functions
            metrics.update(met.__name__, met(out_dict["predict"], out_dict["label"]), n=n)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        # 这里进入训练模式
        self.model.train()
        # self.train_metrics.reset(): 这一行代码调用了一个名为 reset 的方法，该方法可能是用于重置训练过程中的度量（metrics）或统计信息的。这样可以确保每个训练周期（epoch）开始时，度量的状态是干净的，而不会受到之前周期的影响
        self.train_metrics.reset()
        # tqdm 是一个 Python 库，用于在命令行界面中显示进度条，以提供对代码执行进度的实时可视化反馈。它的名称取自阿拉伯语中的“taqaddum”（进展）。
        bar = tqdm(enumerate(self.data_loader), total=len(self.data_loader))
        for batch_idx, batch_dict in bar:
            self.optimizer.zero_grad()  # setup gradient to zero
            # text = self.data_loader.text[self.data_loader.batch_size * (batch_idx): self.data_loader.batch_size * (batch_idx + 1)]
            out_dict = self.run_model(batch_dict, self.model)  # run model
            out_dict["loss"].backward()  # backpropagation
            self.optimizer.step()  # gradient descent
            # self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, "train")
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, "train")
            self.update_metrics(self.train_metrics, out_dict)
            if batch_idx % self.log_step == 0:  # set bar
                bar.set_description(f"Train Epoch: {epoch} Loss: {out_dict['loss'].item()}")
            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        # if self.do_validation:
        #     log.update(self.evaluate(self.valid_loader, self.model, epoch))  # update validation log
        # log.update(self.evaluate(self.test_loader,self.model,epoch,prefix="test"))
        log.update(self.test(self.model,self.d_loader))
        if self.lr_scheduler is not None:
            # 是否调整 lr
            self.lr_scheduler.step()
        return log

    def evaluate(self, loader, model, epoch=0, prefix="val"):
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(loader), total=len(loader)):
                out_dict = self.run_model(batch_dict, model)
                self.writer.set_step((epoch - 1) * len(loader) + batch_idx, "evaluate")
                self.update_metrics(self.valid_metrics, out_dict)
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins='auto')
        return {f"{prefix}_{k}": v for k, v in self.valid_metrics.result().items()}  # return log with prefix


    def test(self, model, data_loader: NewsDataLoader):
        log = {}
        # run validation
        log.update(self.evaluate(data_loader.valid_loader, model, prefix="val"))
        # run test
        log.update(self.evaluate(data_loader.test_loader, model, prefix="test"))
        return log