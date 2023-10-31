from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    print("查看当前配置文件的trainer：", cfg.trainer_module, cfg.trainer_path)
    trainer = make_trainer(cfg, network)  # 初始化训练器trainer类
    optimizer = make_optimizer(cfg, network)  # 初始化优化器
    scheduler = make_lr_scheduler(cfg, optimizer)  # 设置学习率调度器
    recorder = make_recorder(cfg)  # 初始化训练记录类
    evaluator = make_evaluator(cfg)  # 初始化评估器

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)  # 加载已经训练好的网络模型，获取继续训练时的epoch
    set_lr_scheduler(cfg, scheduler)  # 设置学习率调度器

    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)  # 训练数据加载器
    val_loader = make_data_loader(cfg, is_train=False)  # 验证数据加载器

    for epoch in range(begin_epoch, cfg.train.epoch):  # 开始训练
        recorder.epoch = epoch  # 记录当前训练批次
        if cfg.distributed:  # 判断是否为分布式训练
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        trainer.train(epoch, train_loader, optimizer, recorder)  # 当前训练批次
        scheduler.step()  # 学习率更新

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)  # 保存训练模型

        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)  # 保存训练模型

        if (epoch + 1) % cfg.eval_ep == 0:
            trainer.val(epoch, val_loader, evaluator, recorder)  # 验证当前训练好的模型

    return network


def test(cfg, network):
    trainer = make_trainer(cfg, network)  # 初始化测试器
    val_loader = make_data_loader(cfg, is_train=False)  # 数据加载器
    evaluator = make_evaluator(cfg)  # 初始化评估器
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)  # 加载网络
    trainer.val(epoch, val_loader, evaluator)  # 在trainer中进行推理评估


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def main():
    # cfg为从配置文件中读取的参数
    if cfg.distributed:  # 是否为分布式运行；config.py中默认为false
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)  # 设置GPU or CPU计算
        # pytorch分布式训练
        # backend str/Backend 是通信所用的后端，可以是"ncll" "gloo"或者是一个torch.distributed.Backend类（Backend.GLOO）
        # init_method str 这个URL指定了如何初始化互相通信的进程
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()  # 同步

    print("查看当前配置文件的network：", cfg.network_module, cfg.network_path)
    network = make_network(cfg)  # 根据cfg参数初始化网络
    if args.test:  # 测试模式
        test(cfg, network)
    else:  # 训练模式
        train(cfg, network)


if __name__ == "__main__":
    main()
