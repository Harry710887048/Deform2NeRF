import torch
import torch.nn.functional as F
from lib.config import cfg
from .nerf_net_utils import *
from .. import embedder
from lib.utils.blend_utils import *


class Renderer:
    def __init__(self, net):
        self.net = net  # 获取网络，对应network

    def get_wsampling_points(self, ray_o, ray_d, near, far):
        # calculate the steps for each ray，计算每条光线的步长
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

        if cfg.perturb > 0. and self.net.training:
            # get intervals between samples，获取样本之间的间隔
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(upper)
            z_vals = lower + (upper - lower) * t_rand

        pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]

        return pts, z_vals

    def get_density_color(self, wpts, viewdir, z_vals, raw_decoder):
        """
        wpts: n_batch, n_pixel, n_sample, 3
        viewdir: n_batch, n_pixel, 3
        z_vals: n_batch, n_pixel, n_sample
        """
        n_batch, n_pixel, n_sample = wpts.shape[:3]
        wpts = wpts.view(n_batch * n_pixel * n_sample, -1)  # 改变wpts的形状
        viewdir = viewdir[:, :, None].repeat(1, 1, n_sample, 1).contiguous()
        viewdir = viewdir.view(n_batch * n_pixel * n_sample, -1)

        # calculate dists for the opacity computation，计算距离用于不透明度的运算
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, dists[..., -1:]], dim=2)
        dists = dists.view(n_batch * n_pixel * n_sample)

        ret = raw_decoder(wpts, viewdir, dists)  # 获取network输出值

        return ret

    def get_pixel_value(self, ray_o, ray_d, near, far, occ, batch):
        n_batch = ray_o.shape[0]

        # sampling points for nerf training，获取nerf训练的世界空间采样点
        wpts, z_vals = self.get_wsampling_points(ray_o, ray_d, near, far)
        n_batch, n_pixel, n_sample = wpts.shape[:3]

        # viewing direction, ray_d has been normalized in the dataset
        viewdir = ray_d  # 视角方向

        raw_decoder = lambda wpts_val, viewdir_val, dists_val: self.net(
            wpts_val, viewdir_val, dists_val, batch)

        # compute the color and density，# 计算采样点的密度和颜色
        ret = self.get_density_color(wpts, viewdir, z_vals, raw_decoder)

        # reshape to [num_rays, num_samples along ray, 4]
        n_batch, n_pixel, n_sample = z_vals.shape
        raw = ret['raw'].reshape(-1, n_sample, 4)  # 获取network输出值中的raw，即tpose空间采样点的颜色和密度数据，并改变数据形状
        z_vals = z_vals.view(-1, n_sample)
        ray_d = ray_d.view(-1, 3)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(
            raw, z_vals, cfg.white_bkgd)

        rgb_map = rgb_map.view(n_batch, n_pixel, -1)
        acc_map = acc_map.view(n_batch, n_pixel)
        depth_map = depth_map.view(n_batch, n_pixel)

        ret.update({
            'rgb_map': rgb_map,  # 射线对应像素点颜色
            'acc_map': acc_map,
            'depth_map': depth_map,
            'raw': raw.view(n_batch, -1, 4)
        })

        if 'pbw' in ret:
            pbw = ret['pbw'].view(n_batch, -1, 24)
            ret.update({'pbw': pbw})

        if 'tbw' in ret:
            tbw = ret['tbw'].view(n_batch, -1, 24)
            ret.update({'tbw': tbw})

        if 'sdf' in ret:
            # get pixels that outside the mask or no ray-geometry intersection
            sdf = ret['sdf'].view(n_batch, n_pixel, n_sample)
            min_sdf = sdf.min(dim=2)[0]
            free_sdf = min_sdf[occ == 0]
            free_label = torch.zeros_like(free_sdf)

            with torch.no_grad():
                intersection_mask, _ = get_intersection_mask(sdf, z_vals)
            ind = (intersection_mask == False) * (occ == 1)
            sdf = min_sdf[ind]
            label = torch.ones_like(sdf)

            sdf = torch.cat([sdf, free_sdf])
            label = torch.cat([label, free_label])
            ret.update({
                'msk_sdf': sdf.view(n_batch, -1),
                'msk_label': label.view(n_batch, -1)
            })

        if not rgb_map.requires_grad:
            ret = {k: ret[k].detach().cpu() for k in ret.keys()}

        return ret

    def render(self, batch):  # 开始渲染
        ray_o = batch['ray_o']  # 获取光线原点
        ray_d = batch['ray_d']  # 获取光线方向
        near = batch['near']  # 获取近界
        far = batch['far']  # 获取远界
        occ = batch['occupancy']  # 获取人体占用像素集合
        sh = ray_o.shape  # 获取光线原点坐标数据的形状

        # volume rendering for each pixel，对每个像素进行体渲染
        n_batch, n_pixel = ray_o.shape[:2]  # 获取batch数量和像素数量
        chunk = 2048  # 设置chunk值进行批量处理
        ret_list = []
        for i in range(0, n_pixel, chunk):
            ray_o_chunk = ray_o[:, i:i + chunk]
            ray_d_chunk = ray_d[:, i:i + chunk]
            near_chunk = near[:, i:i + chunk]
            far_chunk = far[:, i:i + chunk]
            occ_chunk = occ[:, i:i + chunk]
            pixel_value = self.get_pixel_value(ray_o_chunk, ray_d_chunk,
                                               near_chunk, far_chunk,
                                               occ_chunk, batch)  # 获取像素值
            ret_list.append(pixel_value)

        keys = ret_list[0].keys()
        ret = {k: torch.cat([r[k] for r in ret_list], dim=1) for k in keys}

        return ret
