import numpy as np
from lib.config import cfg
from skimage.measure import compare_ssim
import os
import cv2


class Evaluator:
    def __init__(self):
        self.mse = []
        self.psnr = []
        self.ssim = []

    def psnr_metric(self, img_pred, img_gt):  # 计算峰值信噪比指标
        mse = np.mean((img_pred - img_gt)**2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr

    def ssim_metric(self, rgb_pred, rgb_gt, batch):
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        # H, W = int(cfg.H * cfg.ratio), int(cfg.W * cfg.ratio)
        H = batch['H']  # 获取图片的高
        W = batch['W']  # 获取图片的宽
        mask_at_box = mask_at_box.reshape(H, W)
        # convert the pixels into an image，将像素转换为图像
        img_pred = np.zeros((H, W, 3))  # 设置指定大小的全零矩阵
        img_pred[mask_at_box == 1] = rgb_pred  # 为img_pred指定位置元素矩阵赋值
        img_gt = np.zeros((H, W, 3))  # 设置指定大小的全零矩阵
        img_gt[mask_at_box == 1] = rgb_gt  # 为img_gt指定位置元素矩阵赋值

        result_dir = os.path.join(cfg.result_dir, 'comparison')  # 设置结果保存路径
        os.system('mkdir -p {}'.format(result_dir))  # 创建结果保存路径
        frame_index = batch['frame_index'].item()  # 获取帧索引
        view_index = batch['cam_ind'].item()  # 获取相机【视角】索引
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}.png'.format(result_dir, frame_index,
                                                   view_index),
            (img_pred[..., [2, 1, 0]] * 255))  # 写入图片到指定路径
        cv2.imwrite(
            '{}/frame{:04d}_view{:04d}_gt.png'.format(result_dir, frame_index,
                                                      view_index),
            (img_gt[..., [2, 1, 0]] * 255))  # 写入图片到指定路径

        # crop the object region，裁剪对象区域
        x, y, w, h = cv2.boundingRect(mask_at_box.astype(np.uint8))
        img_pred = img_pred[y:y + h, x:x + w]
        img_gt = img_gt[y:y + h, x:x + w]

        # compute the ssim，计算图片的结构相似指标
        ssim = compare_ssim(img_pred, img_gt, multichannel=True)
        return ssim

    def evaluate(self, output, batch):
        img_pred = output['rgb'][0].permute(1, 2, 0).detach().cpu().numpy()  # 获取预测的图片
        mask = output['mask'][0, 0].detach().cpu().numpy()  # 获取mask
        img_pred[mask < 0.5] = 0  # 令mask中小于0.5的元素值得到一个bool矩阵，并令预测图片中对应位置为true的元素值为0
        img_gt = batch['img'][0].permute(1, 2, 0).detach().cpu().numpy()  # 获取真实图片的矩阵
        if img_gt.sum() == 0:  # 若img_gt的元素值全为0时
            return
        mask_at_box = batch['mask_at_box'][0].detach().cpu().numpy()
        rgb_pred = img_pred[mask_at_box == 1]  # 获取rgb预测值
        rgb_gt = img_gt[mask_at_box == 1]  # 获取rgb真实值

        mse = np.mean((rgb_pred - rgb_gt)**2)  # 计算rgb预测值与真实值的mse
        self.mse.append(mse)  # 保存当前batch的

        psnr = self.psnr_metric(rgb_pred, rgb_gt)  # 计算rgb预测值与真实值的psnr峰值信噪比
        self.psnr.append(psnr)

        ssim = self.ssim_metric(rgb_pred, rgb_gt, batch)  # 计算rgb预测值与真实值的ssim结构相似
        self.ssim.append(ssim)

    def summarize(self):
        result_path = os.path.join(cfg.result_dir, 'metrics.npy')   # 设置结果保存路径
        os.system('mkdir -p {}'.format(os.path.dirname(result_path)))  # 创建result_path路径
        metrics = {'mse': self.mse, 'psnr': self.psnr, 'ssim': self.ssim}  # 设置指标结果字典
        np.save(result_path, metrics)  # 保存指标结果到指定路径
        print('mse: {}'.format(np.mean(self.mse)))  # 输出mse结果
        print('psnr: {}'.format(np.mean(self.psnr)))  # 输出psnr结果
        print('ssim: {}'.format(np.mean(self.ssim)))  # 输出ssim结果
        self.mse = []
        self.psnr = []
        self.ssim = []
