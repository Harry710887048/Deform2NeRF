import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch
from lib.config import cfg
from lib.utils.blend_utils import *
from .. import embedder
from lib.utils import net_utils
from lib.utils import sample_utils
from ..hash_code import hash_embedder
from ..hash_code.hash_embedder import get_embedder


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.tpose_human = TPoseHuman()

        self.bw_latent = nn.Embedding(cfg.num_train_frame + 1, 128)  # 获取混合权重潜在编码

        self.actvn = nn.ReLU()  # 设置激活函数

        input_ch = 191  # 输入通道，63【位置的位置编码】+128【潜在编码】=191
        D = 8  # 设置网络层数
        W = 256  # 设置卷积层输出通道大小
        self.skips = [4]  # 在第五层包含一个跳跃连接
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
                                  self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)  # 设置跳跃连接层
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

        if cfg.aninerf_animation:
            self.novel_pose_bw = BackwardBlendWeight()

            if 'init_aninerf' in cfg:
                net_utils.load_network(self,
                                       'data/trained_model/deform/' +
                                       cfg.init_aninerf,
                                       strict=False)

    def get_bw_feature(self, pts, ind):
        pts = embedder.xyz_embedder(pts)  # 对采样点的位置进行位置编码处理
        pts = pts.transpose(1, 2)
        latent = self.bw_latent(ind)  # 根据数据集给出的潜在编码索引来改变原始潜在编码的排列
        latent = latent[..., None].expand(*latent.shape, pts.size(2))  # 获取潜在编码
        features = torch.cat((pts, latent), dim=1)  # 联结采样点位置数据和潜在编码
        return features

    def calculate_neural_blend_weights(self, pose_pts, smpl_bw, latent_index):
        features = self.get_bw_feature(pose_pts, latent_index)  # 潜在编码索引由数据集提供，获取神经混合权重场的输入
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)  # 设置跳跃层的输入
        bw = self.bw_fc(net)  # 神经混合权重场输出的混合权重
        bw = torch.log(smpl_bw + 1e-9) + bw  # 对应公式5
        bw = F.softmax(bw, dim=1)  # 包含了公式5中的指数映射和归一化处理
        return bw

    def pose_points_to_tpose_points(self, pose_pts, batch):
        """
        pose_pts: n_batch, n_point, 3
        """
        # initial blend weights of points at i，获取姿势【观测】空间点i的初始混合权重【smpl权重】
        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        init_pbw = init_pbw[:, :24]

        # neural blend weights of points at i，获取姿势【观测】空间点i的神经混合权重
        if cfg.test_novel_pose:
            pbw = self.novel_pose_bw(pose_pts, init_pbw,
                                     batch['bw_latent_index'])
        else:
            pbw = self.calculate_neural_blend_weights(
                pose_pts, init_pbw, batch['latent_index'] + 1)  # 获取观察【姿势】空间的混合权重

        # transform points from i to i_0，将点从i转换为i_0，即将姿势【观测】空间的点i转换到tpose【标准】空间中定义为i_0
        tpose = pose_points_to_tpose_points(pose_pts, pbw, batch['A'])

        return tpose, pbw

    # NOTE: this part should actually have been deprecated...
    # we leave this here for reproducability, in the extended version, we implmented a better aninerf pipeline (same core idea as the paper)
    # thus some of the old config files or code could not run as expected especially when outside the core training loop
    def calculate_alpha(self, wpts, batch):
        # transform points from the world space to the pose space
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                            batch['pbounds'])
        pnorm = init_pbw[:, 24]
        norm_th = 0.1
        pind = pnorm < norm_th
        pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
        pose_pts = pose_pts[pind][None]

        # transform points from the pose space to the tpose space
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)

        alpha = self.tpose_human.calculate_alpha(tpose)
        alpha = alpha[0, 0]

        n_batch, n_point = wpts.shape[:2]
        full_alpha = torch.zeros([n_point]).to(wpts)
        full_alpha[pind[0]] = alpha

        return full_alpha

    get_alpha = calculate_alpha

    def forward(self, wpts, viewdir, dists, batch):
        # transform points from the world space to the pose space，将点从世界空间变换到姿势空间【观测空间】
        wpts = wpts[None]
        pose_pts = world_points_to_pose_points(wpts, batch['R'], batch['Th'])

        with torch.no_grad():
            init_pbw = pts_sample_blend_weights(pose_pts, batch['pbw'],
                                                batch['pbounds'])  # 获取采姿势空间采样点的混合权重【数据集中提供了混合权重pbw】
            pnorm = init_pbw[:, -1]
            norm_th = cfg.norm_th
            pind = pnorm < norm_th
            pind[torch.arange(len(pnorm)), pnorm.argmin(dim=1)] = True
            pose_pts = pose_pts[pind][None]
            viewdir = viewdir[pind[0]]
            dists = dists[pind[0]]

        # transform points from the pose space to the tpose space，将点从姿势空间变换到tpose空间【标准空间】
        tpose, pbw = self.pose_points_to_tpose_points(pose_pts, batch)

        # calculate neural blend weights of points at the tpose space
        init_tbw = pts_sample_blend_weights(tpose, batch['tbw'],
                                            batch['tbounds'])  # 计算tpose【标准】空间中点的神经混合权重
        init_tbw = init_tbw[:, :24]
        ind = torch.zeros_like(batch['latent_index'])
        tbw = self.calculate_neural_blend_weights(tpose, init_tbw, ind)  # 计算tpose空间的神经混合权重

        viewdir = viewdir[None]
        ind = batch['latent_index']
        # alpha, rgb = self.tpose_human.calculate_alpha_rgb(tpose, viewdir, ind)  # 计算tpose空间采样点的密度和颜色
        cfg.bounding_box = (batch['tbounds'][0][0], batch['tbounds'][0][0])
        alpha, rgb = self.tpose_human.calculate_alpha_rgb(tpose, viewdir, ind, batch)

        inside = tpose > batch['tbounds'][:, :1]
        inside = inside * (tpose < batch['tbounds'][:, 1:])
        outside = torch.sum(inside, dim=2) != 3
        alpha = alpha[:, 0]
        alpha[outside] = 0

        alpha_ind = alpha.detach() > cfg.train_th
        max_ind = torch.argmax(alpha, dim=1)
        alpha_ind[torch.arange(alpha.size(0)), max_ind] = True
        pbw = pbw.transpose(1, 2)[alpha_ind][None]
        tbw = tbw.transpose(1, 2)[alpha_ind][None]

        raw2alpha = lambda raw, dists, act_fn=F.relu: 1. - torch.exp(-act_fn(
            raw) * dists)

        rgb = torch.sigmoid(rgb[0])
        alpha = raw2alpha(alpha[0], dists)

        raw = torch.cat((rgb, alpha[None]), dim=0)  # 合并采样点的rgb颜色值和alpha密度值
        raw = raw.transpose(0, 1)

        n_batch, n_point = wpts.shape[:2]
        raw_full = torch.zeros([n_batch, n_point, 4], dtype=wpts.dtype, device=wpts.device)  # 定义一个指定大小的零张量
        raw_full[pind] = raw  # 将rbg、alpha值保存到raw_full

        ret = {'pbw': pbw, 'tbw': tbw, 'raw': raw_full}

        return ret


# class TPoseHuman(nn.Module):
#     def __init__(self):
#         super(TPoseHuman, self).__init__()
#
#         self.nf_latent = nn.Embedding(cfg.num_train_frame, 128)  # 定义颜色场网络的潜在编码
#
#         self.actvn = nn.ReLU()
#
#         input_ch = 63
#         D = 8
#         W = 256
#         self.skips = [4]  # 设置跳跃连接层
#         self.pts_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
#             nn.Conv1d(W, W, 1) if i not in
#                                   self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
#         ])
#         self.alpha_fc = nn.Conv1d(W, 1, 1)
#
#         self.feature_fc = nn.Conv1d(W, W, 1)
#         self.latent_fc = nn.Conv1d(384, W, 1)
#         self.view_fc = nn.Conv1d(283, W // 2, 1)
#         self.rgb_fc = nn.Conv1d(W // 2, 3, 1)
#
#     def calculate_alpha(self, nf_pts):
#         nf_pts = embedder.xyz_embedder(nf_pts)
#         input_pts = nf_pts.transpose(1, 2)
#         net = input_pts
#         for i, l in enumerate(self.pts_linears):
#             net = self.actvn(self.pts_linears[i](net))
#             if i in self.skips:
#                 net = torch.cat((input_pts, net), dim=1)  # 设置跳跃层输入
#         alpha = self.alpha_fc(net)
#         return alpha
#
#     def calculate_alpha_rgb(self, nf_pts, viewdir, ind, batch=None):
#         # nf_pts = embedder.xyz_embedder(nf_pts)  # 对tpose空间采样点的位置使用位置编码
#         xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res, args=cfg, i=1)  # arg表示hash编码的参数
#         view_embedder, view_dim = get_embedder(cfg.view_res, args=cfg, i=2)
#         nf_pts = xyz_embedder(nf_pts)
#         input_pts = nf_pts.transpose(1, 2)
#         net = input_pts
#         for i, l in enumerate(self.pts_linears):
#             net = self.actvn(self.pts_linears[i](net))
#             if i in self.skips:
#                 net = torch.cat((input_pts, net), dim=1)  # 设置跳跃层输入
#         alpha = self.alpha_fc(net)  # 输出tpose空间采样点的密度
#
#         features = self.feature_fc(net)
#
#         latent = self.nf_latent(ind)
#         latent = latent[..., None].expand(*latent.shape, net.size(2))  # 获取潜在编码l
#         features = torch.cat((features, latent), dim=1)  # 联结网络层特征和潜在编码l
#         features = self.latent_fc(features)
#
#         # viewdir = embedder.view_embedder(viewdir)  # 对tpose空间采样点的方向使用位置编码
#         viewdir = view_embedder(viewdir)
#         viewdir = viewdir.transpose(1, 2)
#         features = torch.cat((features, viewdir), dim=1)  # 联结网络层特征和采样点方向的位置编码
#         net = self.actvn(self.view_fc(features))
#         rgb = self.rgb_fc(net)  # 输出tpose空间采样点的颜色
#
#         return alpha, rgb

# Small NeRF for Hash embeddings
class TPoseHuman(nn.Module):
    def __init__(self,
                 num_layers=3,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=4,
                 hidden_dim_color=64,
                 input_ch=32, input_ch_views=16,
                 ):
        super(TPoseHuman, self).__init__()

        self.input_ch = input_ch
        self.input_ch_views = input_ch_views

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.input_ch
            else:
                in_dim = hidden_dim

            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim  # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim

            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color
        self.hidden_dim_color = hidden_dim_color

        color_net = []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.input_ch_views + self.geo_feat_dim
            else:
                in_dim = hidden_dim

            if l == num_layers_color - 1:
                out_dim = 3  # 3 rgb
            else:
                out_dim = hidden_dim

            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

    def calculate_alpha_rgb(self, nf_pts, viewdir, ind, batch=None):
        xyz_embedder, xyz_dim = get_embedder(cfg.xyz_res, args=cfg, i=1)  # arg表示hash编码的参数
        view_embedder, view_dim = get_embedder(cfg.view_res, args=cfg, i=2)

        nf_pts = xyz_embedder(nf_pts)
        viewdir = view_embedder(viewdir)  # 对tpose空间采样点的方向使用球谐编码

        input_pts = nf_pts.transpose(1, 2)
        input_views = viewdir.transpose(1, 2)

        # sigma
        h = input_pts
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        alpha, geo_feat = h[..., 0], h[..., 1:]

        # color
        h = torch.cat([input_views, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)

        # color = torch.sigmoid(h)
        rgb = h

        return alpha, rgb


class BackwardBlendWeight(nn.Module):
    def __init__(self):
        super(BackwardBlendWeight, self).__init__()

        self.bw_latent = nn.Embedding(cfg.num_eval_frame, 128)

        self.actvn = nn.ReLU()

        input_ch = 191
        D = 8
        W = 256
        self.skips = [4]
        self.bw_linears = nn.ModuleList([nn.Conv1d(input_ch, W, 1)] + [
            nn.Conv1d(W, W, 1) if i not in
                                  self.skips else nn.Conv1d(W + input_ch, W, 1) for i in range(D - 1)
        ])
        self.bw_fc = nn.Conv1d(W, 24, 1)

    def get_point_feature(self, pts, ind, latents):
        pts = embedder.xyz_embedder(pts)
        pts = pts.transpose(1, 2)
        latent = latents(ind)
        latent = latent[..., None].expand(*latent.shape, pts.size(2))
        features = torch.cat((pts, latent), dim=1)
        return features

    def forward(self, ppts, smpl_bw, latent_index):
        latents = self.bw_latent
        features = self.get_point_feature(ppts, latent_index, latents)
        net = features
        for i, l in enumerate(self.bw_linears):
            net = self.actvn(self.bw_linears[i](net))
            if i in self.skips:
                net = torch.cat((features, net), dim=1)
        bw = self.bw_fc(net)
        bw = torch.log(smpl_bw + 1e-9) + bw
        bw = F.softmax(bw, dim=1)
        return bw
