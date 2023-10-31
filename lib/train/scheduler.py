from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR


def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler  # 从parent配置文件获取学习率调度器参数
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    return scheduler  # 返回设置好的调度器


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler  # 从parent配置文件获取学习率调度器参数
    if cfg_scheduler.type == 'multi_step':  # 获取调度器参数
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':  # 获取调度器参数
        scheduler.decay_epochs = cfg_scheduler.decay_epochs  # 设置学习率衰减
    scheduler.gamma = cfg_scheduler.gamma  # 设置初始学习率大小
