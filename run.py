from lib.config import cfg, args


def run_dataset():
    from lib.datasets import make_data_loader
    import tqdm

    cfg.train.num_workers = 0
    data_loader = make_data_loader(cfg, is_train=False)
    for batch in tqdm.tqdm(data_loader):
        pass


def run_network():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    import tqdm
    import torch
    import time

    network = make_network(cfg).cuda()
    load_network(network, cfg.trained_model_dir, epoch=cfg.test.epoch)
    network.eval()

    data_loader = make_data_loader(cfg, is_train=False)
    total_time = 0
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            torch.cuda.synchronize()
            start = time.time()
            network(batch)
            torch.cuda.synchronize()
            total_time += time.time() - start
    print(total_time / len(data_loader))


def run_evaluate():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    import torch
    from lib.networks import make_network
    from lib.utils import net_utils
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0  # 设置扰动值
    cfg.eval = True  # 设置eval模式为True

    print("查看当前配置文件的network：", cfg.network_module, cfg.network_path)
    network = make_network(cfg).cuda()  # 获取网络并加载到cuda
    net_utils.load_network(network,
                           cfg.trained_model_dir,
                           resume=cfg.resume,
                           epoch=cfg.test.epoch)  # 加载已训练的模型
    network.train()  # 设置网络训练模式

    data_loader = make_data_loader(cfg, is_train=False)  # 获取数据加载器
    print("查看当前配置文件的renderer：", cfg.renderer_module, cfg.renderer_path)
    renderer = make_renderer(cfg, network)
    print("查看当前配置文件的evaluator：", cfg.evaluator_path, cfg.evaluator_module)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):  # 遍历batch
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
        evaluator.evaluate(output, batch)
    evaluator.summarize()


def run_visualize():
    from lib.networks import make_network
    from lib.datasets import make_data_loader
    from lib.utils.net_utils import load_network
    from lib.utils import net_utils
    import tqdm
    import torch
    from lib.visualizers import make_visualizer
    from lib.networks.renderer import make_renderer

    cfg.perturb = 0

    network = make_network(cfg).cuda()
    load_network(network,
                 cfg.trained_model_dir,
                 resume=cfg.resume,
                 epoch=cfg.test.epoch,
                 strict=False)
    network.train()

    data_loader = make_data_loader(cfg, is_train=False)
    renderer = make_renderer(cfg, network)
    visualizer = make_visualizer(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        with torch.no_grad():
            output = renderer.render(batch)
            visualizer.visualize(output, batch)


def run_light_stage():
    from lib.utils.light_stage import ply_to_occupancy
    ply_to_occupancy.ply_to_occupancy()
    # ply_to_occupancy.create_voxel_off()


def run_evaluate_nv():
    from lib.datasets import make_data_loader
    from lib.evaluators import make_evaluator
    import tqdm
    from lib.utils import net_utils

    data_loader = make_data_loader(cfg, is_train=False)
    evaluator = make_evaluator(cfg)
    for batch in tqdm.tqdm(data_loader):
        for k in batch:
            if k != 'meta':
                batch[k] = batch[k].cuda()
        evaluator.evaluate(batch)
    evaluator.summarize()


def run_animation():
    from tools import animate_mesh
    animate_mesh.animate()


def run_raster():
    from tools import rasterizer_mesh
    renderer = rasterizer_mesh.Renderer()
    renderer.render()


def run_lpips():
    from tools import calculate_lpips
    calculate_lpips.run()


if __name__ == '__main__':
    globals()['run_' + args.type]()  # 根据所给参数调用指定函数
