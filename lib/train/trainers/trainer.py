import time
import datetime
import torch
import tqdm
from torch.nn import DataParallel
from lib.config import cfg


class Trainer(object):
    def __init__(self, network):
        device = torch.device('cuda:{}'.format(cfg.local_rank))  # 设置训练device
        network = network.to(device)  # 将网络放到device上
        if cfg.distributed:  # 判断是否为分布式训练
            network = torch.nn.parallel.DistributedDataParallel(
                network,
                device_ids=[cfg.local_rank],
                output_device=cfg.local_rank
            )
        self.network = network  # 设置网络
        self.local_rank = cfg.local_rank
        self.device = device  # 设置device

    def reduce_loss_stats(self, loss_stats):
        reduced_losses = {k: torch.mean(v) for k, v in loss_stats.items()}
        return reduced_losses

    def to_cuda(self, batch):  # 将张量数据放到cuda上
        if isinstance(batch, tuple) or isinstance(batch, list):
            batch = [self.to_cuda(b) for b in batch]
            return batch

        for k in batch:
            if k == 'meta':
                continue
            if isinstance(batch[k], tuple) or isinstance(batch[k], list):
                batch[k] = [b.to(self.device) for b in batch[k]]
            else:
                batch[k] = batch[k].to(self.device)

        return batch

    def add_iter_step(self, batch, iter_step):
        if isinstance(batch, tuple) or isinstance(batch, list):
            for batch_ in batch:
                self.add_iter_step(batch_, iter_step)  # 递归遍历到batch的最后一个元素

        if isinstance(batch, dict):
            batch['iter_step'] = iter_step

    def train(self, epoch, data_loader, optimizer, recorder):  # 开始训练
        max_iter = len(data_loader)  # 获取数据大小，即一个epoch的data_loader循环次数
        self.network.train()  # 将网络设置为训练模式
        end = time.time()
        for iteration, batch in enumerate(data_loader):  # 开始遍历数据加载器
            data_time = time.time() - end
            iteration = iteration + 1

            batch = self.to_cuda(batch)  # 将数据存放到cuda上
            self.add_iter_step(batch, epoch * max_iter + iteration)
            output, loss, loss_stats, image_stats = self.network(batch)  # 获取输出、损失、损失统计、图片统计【在配置文件中指定的trainer中进行前向传播】

            # training stage: loss; optimizer; scheduler
            if cfg.training_mode == 'default':
                optimizer.zero_grad()  # 梯度清零
                loss = loss.mean()  # 获取损失均值
                loss.backward()  # 反向传播
                torch.nn.utils.clip_grad_value_(self.network.parameters(), 40)  # 梯度裁剪
                optimizer.step()  # 参数更新
            else:
                optimizer.step()
                optimizer.zero_grad()

            if cfg.local_rank > 0:
                continue

            # data recording stage: loss_stats, time, image_stats
            recorder.step += 1

            loss_stats = self.reduce_loss_stats(loss_stats)
            recorder.update_loss_stats(loss_stats)

            batch_time = time.time() - end
            end = time.time()
            recorder.batch_time.update(batch_time)
            recorder.data_time.update(data_time)

            if iteration % cfg.log_interval == 0 or iteration == (max_iter - 1):
                # print training state
                eta_seconds = recorder.batch_time.global_avg * (max_iter - iteration)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                lr = optimizer.param_groups[0]['lr']
                memory = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0  # 获取当前内存占用情况

                exp_name = 'exp: {}'.format(cfg.exp_name)
                training_state = '  '.join([exp_name, 'eta: {}', '{}', 'lr: {:.6f}', 'max_mem: {:.0f}'])
                training_state = training_state.format(eta_string, str(recorder), lr, memory)
                print(training_state)  # 打印训练结果

            if iteration % cfg.record_interval == 0 or iteration == (max_iter - 1):
                # record loss_stats and image_dict
                recorder.update_image_stats(image_stats)
                recorder.record('train')

    def val(self, epoch, data_loader, evaluator=None, recorder=None):
        self.network.eval()  # 将网络设置为评估模式
        torch.cuda.empty_cache()  # PyTorch的缓存分配器会事先分配一些固定的显存，即使实际上tensors并没有使用完这些显存，这些显存也不能被其他应用使用。这个分配过程由第一次CUDA内存访问触发的。
        val_loss_stats = {}  # 验证损失统计
        data_size = len(data_loader)  # 获取data_loader划分的batch块数量
        for batch in tqdm.tqdm(data_loader):  # 遍历batch块
            batch = self.to_cuda(batch)  # 将batch转换到cuda上运算
            with torch.no_grad():
                output, loss, loss_stats, image_stats = self.network(batch)  # 在配置文件中指定的trainer中进行前向传播，获取输出和损失
                if evaluator is not None:
                    evaluator.evaluate(output, batch)

            loss_stats = self.reduce_loss_stats(loss_stats)
            for k, v in loss_stats.items():
                val_loss_stats.setdefault(k, 0)
                val_loss_stats[k] += v

        loss_state = []
        for k in val_loss_stats.keys():
            val_loss_stats[k] /= data_size
            loss_state.append('{}: {:.4f}'.format(k, val_loss_stats[k]))
        print(loss_state)

        if evaluator is not None:
            result = evaluator.summarize()
            val_loss_stats.update(result)

        if recorder:
            recorder.record('val', epoch, val_loss_stats, image_stats)
