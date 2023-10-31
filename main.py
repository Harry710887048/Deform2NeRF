import os

# 在训练好的人体姿势上进行测试
# os.system('python run.py --type evaluate --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume True')
# 查看当前配置文件的network： lib.networks.bw_deform.tpose_nerf_network lib/networks/bw_deform/tpose_nerf_network.py

# 查看当前配置文件的trainer:  lib.train.trainers.tpose_trainer lib/train/trainers/tpose_trainer.py
# 查看当前配置文件的renderer： lib.networks.renderer.tpose_renderer lib/networks/renderer/tpose_renderer.py

# 查看当前配置文件的dataset： lib.datasets.tpose_dataset lib/datasets/tpose_dataset.py
'''
batch.keys = dict_keys(['rgb', 'occupancy', 'ray_o', 'ray_d', 'near', 'far', 'mask_at_box', 'A', 'big_A', 
'pbw', 'tbw', 'pbounds', 'wbounds', 'tbounds', 'R', 'Th', 'H', 'W', 'latent_index', 'bw_latent_index', 
'frame_index', 'cam_ind'])
rgb torch.Size([1, 72381, 3])
occupancy torch.Size([1, 72381])  【72381表示当前图片人体占用部分的像素点集合】
ray_o torch.Size([1, 72381, 3])
ray_d torch.Size([1, 72381, 3])
near torch.Size([1, 72381])
far torch.Size([1, 72381])
mask_at_box torch.Size([1, 262144])  【262144=H*W=512*512=1024*0.5*1024*0.5】
A torch.Size([1, 24, 4, 4])  【24表示将人体划分为24个部分，详见SMPL模型定义的24个关节点】
big_A torch.Size([1, 24, 4, 4])
pbw torch.Size([1, 63, 75, 38, 25])
tbw torch.Size([1, 76, 75, 19, 25])
pbounds torch.Size([1, 2, 3])  【保存了3D边界框的最小顶点和最大顶点，下同】
wbounds torch.Size([1, 2, 3])
tbounds torch.Size([1, 2, 3])
R torch.Size([1, 3, 3])
Th torch.Size([1, 1, 3])
H torch.Size([1]) 【根据原始宽高和比率计算】
W torch.Size([1]) 【根据原始宽高和比率计算】
latent_index torch.Size([1])
bw_latent_index torch.Size([1])
frame_index torch.Size([1])
cam_ind torch.Size([1])
'''
# 注：
#     lbs_weights:  6890 * 24 混合权重矩阵，即关节点对顶点的影响权重 (第几个顶点受哪些关节点的影响且权重分别为多少) 6890个顶点，
#     每一个顶点受到24个关节点的影响

# 训练
os.system("python train_net.py --cfg_file configs/aninerf_313.yaml exp_name aninerf_313 resume False")
# 查看当前配置文件的network： lib.networks.bw_deform.tpose_nerf_network lib/networks/bw_deform/tpose_nerf_network.py
# 查看当前配置文件的trainer： lib.train.trainers.tpose_trainer lib/train/trainers/tpose_trainer.py
# 查看当前配置文件的renderer： lib.networks.renderer.tpose_renderer lib/networks/renderer/tpose_renderer.py
# 查看当前配置文件的dataset： lib.datasets.tpose_dataset lib/datasets/tpose_dataset.py

