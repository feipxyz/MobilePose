'''
File: dataset_factory.py
Project: MobilePose-PyTorch
File Created: Sunday, 10th March 2019 8:02:12 pm
Author: Yuliang Xiu (yuliangxiu@sjtu.edu.cn)
-----
Last Modified: Monday, 11th March 2019 12:51:11 am
Modified By: Yuliang Xiu (yuliangxiu@sjtu.edu.cn>)
-----
Copyright 2018 - 2019 Shanghai Jiao Tong University, Machine Vision and Intelligence Group
'''


from dataloader import Rescale, Wrap, PoseDataset, ToTensor, Augmentation, Expansion
from torchvision import datasets, transforms, utils, models
import os

def get_transform(input_size):
    """
    :param modeltype: "resnet" / "mobilenet"
    :param input_size:
    :return:
    """
    return Rescale((input_size, input_size))


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_train_dataset(cfg):
        """
        :param modeltype: "resnet" / "mobilenet"
        :return: type: PoseDataset
        Example:
        DataFactory.get_train_dataset("resnet", 224)
        In debug mode, it will return a small dataset
        """
        return PoseDataset(
            csv_file= os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_LABEL_FILE),
            transform=transforms.Compose([
                Augmentation(),
                get_transform(cfg.DATASET.INPUT_SIZE),
            #    Expansion(),
                ToTensor()
            ]))

    @staticmethod
    def get_test_dataset(cfg):
        """
        :param modeltype: resnet / mobilenet
        :return: type: PoseDataset
        Example:
        DataFactory.get_test_dataset("resnet", 224)
        In debug mode, it will return a small dataset
        """
        return PoseDataset(
            csv_file= os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TEST_LABEL_FILE),
            transform=transforms.Compose([
                get_transform(cfg.DATASET.INPUT_SIZE),
                # Expansion(),
                ToTensor()
            ]))
