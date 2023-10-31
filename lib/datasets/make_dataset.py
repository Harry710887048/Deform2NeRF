from .transforms import make_transforms
from . import samplers
import torch
import torch.utils.data
import imp
import os
from .collate_batch import make_collator
import numpy as np
import time
from lib.config.config import cfg
import cv2

torch.multiprocessing.set_sharing_strategy('file_system')


def _dataset_factory(is_train):
    if is_train:
        module = cfg.train_dataset_module  # 获取parent_cfg对应配置文件中的train_dataset_module
        path = cfg.train_dataset_path  # module的对应路径
        args = cfg.train_dataset  # parent_cfg对应配置文件中训练数据集的参数
    else:
        module = cfg.test_dataset_module
        path = cfg.test_dataset_path
        args = cfg.test_dataset
    print("查看当前配置文件的dataset：", module, path)
    dataset = imp.load_source(module, path).Dataset(**args)  # 加载train_dataset_module类，使用args参数初始化
    return dataset


def make_dataset(cfg, dataset_name, transforms, is_train=True):
    dataset = _dataset_factory(is_train)
    return dataset


def make_data_sampler(dataset, shuffle, is_distributed, is_train):
    if not is_train and cfg.test.sampler == 'FrameSampler':
        sampler = samplers.FrameSampler(dataset)
        return sampler
    if is_distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def make_batch_data_sampler(cfg, sampler, batch_size, drop_last, max_iter,
                            is_train):
    if is_train:
        batch_sampler = cfg.train.batch_sampler
        sampler_meta = cfg.train.sampler_meta
    else:
        batch_sampler = cfg.test.batch_sampler
        sampler_meta = cfg.test.sampler_meta

    if batch_sampler == 'default':
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, batch_size, drop_last)
    elif batch_sampler == 'image_size':
        batch_sampler = samplers.ImageSizeBatchSampler(sampler, batch_size,
                                                       drop_last, sampler_meta)

    if max_iter != -1:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, max_iter)
    return batch_sampler


def worker_init_fn(worker_id):
    cv2.setNumThreads(1)  # MARK: OpenCV undistort is why all cores are taken
    np.random.seed(worker_id + (int(round(time.time() * 1000) % (2**16))))


def make_data_loader(cfg, is_train=True, is_distributed=False, max_iter=-1):
    if is_train:  # 设置训练时的参数
        batch_size = cfg.train.batch_size  # 设置训练批次大小
        # shuffle = True
        shuffle = cfg.train.shuffle  # 设置是否打乱数据集
        drop_last = False
    else:
        batch_size = cfg.test.batch_size
        shuffle = True if is_distributed else False
        drop_last = False

    dataset_name = cfg.train.dataset if is_train else cfg.test.dataset  # 设置数据名称，源于lib/config/config.py文件

    transforms = make_transforms(cfg, is_train)  # 设置图像增强，张量转换及像素归一化
    dataset = make_dataset(cfg, dataset_name, transforms, is_train)
    sampler = make_data_sampler(dataset, shuffle, is_distributed, is_train)  # 设置采样器
    batch_sampler = make_batch_data_sampler(cfg, sampler, batch_size,
                                            drop_last, max_iter, is_train)  # 批次采样器
    num_workers = cfg.train.num_workers
    collator = make_collator(cfg, is_train)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_sampler=batch_sampler,
                                              num_workers=num_workers,
                                              collate_fn=collator,
                                              worker_init_fn=worker_init_fn)  # 获取数据加载器

    return data_loader
