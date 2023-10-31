import os

import cv2
import imageio
import numpy as np
import pandas as pd
import torch.utils.data as data

from lib.config.config import cfg
from lib.utils.if_nerf import if_nerf_data_utils as if_nerf_dutils


class Dataset(data.Dataset):
    def __init__(self, data_root, human, ann_file, split):
        super(Dataset, self).__init__()

        self.data_root = data_root  # 数据存放路径
        self.human = human  # 数据集名称
        self.split = split  # 数据集划分方式

        annots = np.load(ann_file, allow_pickle=True).item()  # annots.npy为数据对应的标注数据
        self.cams = annots['cams']  # 获取相机数据，其中包含K、R、T、D四个参数，分别表示所有帧的所有视角下图片的相机参数、旋转矩阵、平移矩阵、畸变系数

        num_cams = len(self.cams['K'])  # 获取相机数量
        if len(cfg.test_view) == 0:  # 获取test_view的数量
            test_view = [
                i for i in range(num_cams) if i not in cfg.training_view
            ]  # 将非训练视角以外的相机视角设为test_view
            if len(test_view) == 0:
                test_view = [0]  # 设置第0个相机视角为测试视角
        else:
            test_view = cfg.test_view
        view = cfg.training_view if split == 'train' else test_view  # 获取视角view【yaml配置文件设置了4个训练视角】

        i = cfg.begin_ith_frame  # 获取起始帧
        i_intv = cfg.frame_interval  # 获取帧间隔
        ni = cfg.num_train_frame  # 获取训练帧数量
        if cfg.test_novel_pose or cfg.aninerf_animation:  # 当测试新视角或aninerf_animation
            i = cfg.begin_ith_frame + cfg.num_train_frame * i_intv  # 计算新的起始帧
            ni = cfg.num_eval_frame  # 获取评估帧数量

        if self.split == "train":
            df = pd.read_csv(os.path.join(self.data_root, 'splits', 'train.csv'))
            self.ims_inds = np.array(df['frame_index'])
            self.cam_inds = np.array(df['view_index'])
            self.ims = []
            for i1, i2 in zip(self.ims_inds, self.cam_inds):
                self.ims.append(annots['ims'][:][:][i1]['ims'][i2])
            self.ims = np.array(self.ims)
        if self.split == "test":
            df = pd.read_csv(os.path.join(self.data_root, 'splits', 'val_ind.csv'))
            self.ims_inds = np.array(df['frame_index'])
            self.cam_inds = np.array(df['view_index'])
            self.ims = []
            for i1, i2 in zip(self.ims_inds, self.cam_inds):
                self.ims.append(annots['ims'][:][:][i1]['ims'][i2])
            self.ims = np.array(self.ims)
        # print(self.ims, self.cam_inds)
        # exit()
        self.num_cams = len(view)  # 获取相机数量，即视角数量

        self.lbs_root = os.path.join(self.data_root, 'lbs')  # 获取lbs数据路径
        joints = np.load(os.path.join(self.lbs_root, 'joints.npy'))  # 读取lbs数据路径下的joints.npy文件；保存了smpl人体Tpose姿势24关节点数据
        self.joints = joints.astype(np.float32)  # 设置joints数据的数据类型
        self.parents = np.load(
            os.path.join(self.lbs_root, 'parents.npy'))  # 读取lbs数据路径下的parents.npy文件;保存了上述关节点中每一个关节点的父节点
        self.big_A = self.load_bigpose()  # 刚性变换矩阵，大小为24*4*4
        self.nrays = cfg.N_rand  # 光线批处理数量

    def load_bigpose(self):
        big_poses = np.zeros([len(self.joints), 3]).astype(np.float32).ravel()  # 定义一个指定大小和类型的全零矩阵并平铺
        angle = 30
        big_poses[5] = np.deg2rad(angle)  # 度数表示转π表示
        big_poses[8] = np.deg2rad(-angle)
        big_poses = big_poses.reshape(-1, 3)  # 改变big_poses形状
        big_A = if_nerf_dutils.get_rigid_transformation(
            big_poses, self.joints, self.parents)  # 获取刚性变换矩阵
        big_A = big_A.astype(np.float32)  # 修改big_A的数据类型
        return big_A

    def get_mask(self, index):
        msk_path = os.path.join(self.data_root, 'mask_cihp',
                                self.ims[index])[:-4] + '.png'  # 在mask_cihp文件夹下索引图片
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.png'
        if not os.path.exists(msk_path):
            msk_path = os.path.join(self.data_root, self.ims[index].replace(
                'images', 'mask'))[:-4] + '.jpg'
        msk_cihp = imageio.imread(msk_path)  # 读取msk_cihp图片
        if len(msk_cihp.shape) == 3:
            msk_cihp = msk_cihp[..., 0]
        if 'deepcap' in self.data_root:
            msk_cihp = (msk_cihp > 125).astype(np.uint8)
        else:
            msk_cihp = (msk_cihp != 0).astype(np.uint8)
        msk = msk_cihp
        orig_msk = msk.copy()

        if not cfg.eval and cfg.erode_edge:
            border = 5
            kernel = np.ones((border, border), np.uint8)  # 定义一个指定大小的全1矩阵，作为腐蚀膨胀操作的卷积核
            msk_erode = cv2.erode(msk.copy(), kernel)  # 腐蚀
            msk_dilate = cv2.dilate(msk.copy(), kernel)  # 膨胀
            msk[(msk_dilate - msk_erode) == 1] = 100  # 获取轮廓

        return msk, orig_msk

    def prepare_input(self, i):
        # read xyz in the world coordinate system，在世界坐标系中读取xyz
        vertices_path = os.path.join(self.data_root, cfg.vertices,
                                     '{}.npy'.format(i))
        wxyz = np.load(vertices_path).astype(np.float32)  # 从parent_cfg配置文件文件下设置的vertices路径下加载

        # transform smpl from the world coordinate to the smpl coordinate，将smpl从世界坐标转换为smpl坐标
        params_path = os.path.join(self.data_root, cfg.params,
                                   '{}.npy'.format(i))  # 从parent_cfg配置文件文件下设置的params路径下加载
        params = np.load(params_path, allow_pickle=True).item()
        Rh = params['Rh'].astype(np.float32)  # 从params中获取旋转向量
        Th = params['Th'].astype(np.float32)  # 从params中获取平移向量

        # prepare sp input of param pose
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # 旋转向量和矩阵相互转换
        pxyz = np.dot(wxyz - Th, R).astype(np.float32)  # 将世界空间坐标转换到姿势空间

        # calculate the skeleton transformation，计算骨架变换
        poses = params['poses'].reshape(-1, 3)
        joints = self.joints  # 各个关节的位置
        parents = self.parents  # 24 每一个节点的父节点，显然根节点没有父节点
        A = if_nerf_dutils.get_rigid_transformation(poses, joints, parents)  # 获取刚性变换矩阵，即论文中的人体骨骼变换矩阵

        pbw = np.load(os.path.join(self.lbs_root, 'bweights/{}.npy'.format(i)))  # 加载姿势空间的混合权重
        pbw = pbw.astype(np.float32)

        return wxyz, pxyz, A, pbw, Rh, Th, poses

    def __getitem__(self, index):
        img_path = os.path.join(self.data_root, self.ims[index])  # 获取被索引图片的路径
        img = imageio.imread(img_path).astype(np.float32) / 255.  # 读取图片数据，并进行像素归一化
        msk, orig_msk = self.get_mask(index)  # 获取对应图片的掩码数据

        H, W = img.shape[:2]  # 获取图片宽高【1024*1024】
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)  # 使用最近邻插值法改变掩码msk的宽高
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)  # 使用最近邻插值法改变orig_msk的宽高

        cam_ind = self.cam_inds[index]  # 获取当前相机索引
        K = np.array(self.cams['K'][cam_ind])  # 获取当前相机参数
        D = np.array(self.cams['D'][cam_ind])  # 获取当前畸变系数
        img = cv2.undistort(img, K, D)  # 对img进行去畸变处理
        msk = cv2.undistort(msk, K, D)  # 对msk进行去畸变处理
        orig_msk = cv2.undistort(orig_msk, K, D)  # 对orig_msk进行去畸变处理

        R = np.array(self.cams['R'][cam_ind])  # 获取当前相机的旋转矩阵R
        T = np.array(self.cams['T'][cam_ind]) / 1000.  # 获取当前相机的平移向量T

        # reduce the image resolution by ratio，按比例降低图像分辨率
        H, W = int(img.shape[0] * cfg.ratio), int(img.shape[1] * cfg.ratio)  # 512, 512
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)  # 使用基于图像区域方法改变掩码msk的宽高
        msk = cv2.resize(msk, (W, H), interpolation=cv2.INTER_NEAREST)  # 使用最近邻插值法改变掩码msk的宽高
        orig_msk = cv2.resize(orig_msk, (W, H),
                              interpolation=cv2.INTER_NEAREST)  # 使用最近邻插值法改变掩码orig_msk的宽高
        if cfg.mask_bkgd:
            img[msk == 0] = 0  # 设置img的背景为像素值为全0，即完全不透明的白色,也即是无色
        K[:2] = K[:2] * cfg.ratio  # 更新相机参数

        if self.human in ['CoreView_313', 'CoreView_315']:  # 判断数据集名字
            i = int(os.path.basename(img_path).split('_')[4])
            frame_index = i - 1
        else:
            i = int(os.path.basename(img_path)[:-4])
            frame_index = i

        # read v_shaped
        # vertices_path = os.path.join(self.lbs_root, 'bigpose_vertices.npy')
        vertices_path = os.path.join(self.lbs_root, 'tvertices.npy')  # 获取tpose顶点数据路径
        tpose = np.load(vertices_path).astype(np.float32)  # 加载tpose顶点数据
        tbounds = if_nerf_dutils.get_bounds(tpose)  # 根据tpose顶点数据获取tpose的3D边界框
        # tbw = np.load(os.path.join(self.lbs_root, 'bigpose_bw.npy'))
        tbw = np.load(os.path.join(self.lbs_root, 'tbw.npy'))  # 加载tpose空间的混合权重
        tbw = tbw.astype(np.float32)  # 设置tbw的数据类型

        wpts, ppts, A, pbw, Rh, Th, poses = self.prepare_input(
            i)  # 获取世界空间坐标、姿势空间坐标、人体骨骼变换矩阵、姿势空间混合权重、旋转向量、平移向量、smpl人体姿势参数

        pbounds = if_nerf_dutils.get_bounds(ppts)  # 根据姿势空间顶点数据获取姿势空间的3D边界框
        wbounds = if_nerf_dutils.get_bounds(wpts)  # 根据世界空间顶点数据获取世界空间的3D边界框

        rgb, ray_o, ray_d, near, far, coord, mask_at_box = if_nerf_dutils.sample_ray_h36m(
            img, msk, K, R, T, wbounds, self.nrays, self.split)  # 从图片中获取了nrays【1024】个像素和射线

        if cfg.erode_edge:
            orig_msk = if_nerf_dutils.crop_mask_edge(orig_msk)
        occupancy = orig_msk[coord[:, 0], coord[:, 1]]  # 获取图片中人体区域占用像素位置的集合

        # nerf
        ret = {
            'rgb': rgb,  # 图片人体区域像素值集合
            'occupancy': occupancy,
            'ray_o': ray_o,  # 光线原点
            'ray_d': ray_d,  # 光线方向
            'near': near,  # 近界
            'far': far,  # 远界
            'mask_at_box': mask_at_box,  # box处的遮罩
            'poses': poses
        }

        # blend weight，混合权重
        meta = {
            'A': A,  # 表示所有其他节点相对根节点的刚体变换矩阵24*3*3
            'big_A': self.big_A,
            'pbw': pbw,
            'tbw': tbw,
            'pbounds': pbounds,
            'wbounds': wbounds,
            'tbounds': tbounds
        }
        ret.update(meta)  # 更新字典

        # transformation
        '''
        # 把旋转矩阵转化为旋转向量
        rvec, _ = cv2.Rodrigues(rot_mat)
        # 把旋转向量转换为旋转矩阵
        rot_mat, _ = cv2.Rodrigues(rvec)
        '''
        R = cv2.Rodrigues(Rh)[0].astype(np.float32)  # cv2.Rodrigues表示将旋转矩阵和向量相互转化
        meta = {'R': R, 'Th': Th, 'H': H, 'W': W}
        ret.update(meta)  # 更新字典

        latent_index = index // self.num_cams  # 索引index对相机数量向下取整除，返回整除结果的整数部分
        bw_latent_index = index // self.num_cams  # 同上
        if cfg.test_novel_pose:  # 判断是否测试新视角姿势
            if 'h36m' in self.data_root:  # 判断是否在h36m上测试
                latent_index = 0  # 设置潜在索引值
            else:
                latent_index = cfg.num_train_frame - 1  # 设置潜在索引
        meta = {
            'latent_index': latent_index,
            'bw_latent_index': bw_latent_index,
            'frame_index': frame_index,
            'cam_ind': cam_ind
        }
        ret.update(meta)  # 更新字典

        return ret  # 返回数据

    def __len__(self):
        return len(self.ims)
