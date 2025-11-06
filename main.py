"""
# Created: 2024-11-20 13:11
# Copyright (C) 2024-now, RPL, KTH Royal Institute of Technology
# Author: Qingwen Zhang  (https://kin-zhang.github.io/)
#
# This file is part of DUFOMap (https://github.com/KTH-RPL/dufomap) and 
# DynamicMap Benchmark (https://github.com/KTH-RPL/DynamicMap_Benchmark) projects.
# If you find this repo helpful, please cite the respective publication as 
# listed on the above website.

# Description: Output Cleaned Map through Python API.
"""
from pathlib import Path
import os, fire, time
import numpy as np
from tqdm import tqdm

from dufomap import dufomap
from dufomap.utils import pcdpy3
def inv_pose_matrix(pose):
    inv_pose = np.eye(4)
    inv_pose[:3, :3] = pose[:3, :3].T
    inv_pose[:3, 3] = -pose[:3, :3].T.dot(pose[:3, 3])
    return inv_pose

MIN_AXIS_RANGE = 0.2 # HARD CODED: remove ego vehicle points
MAX_AXIS_RANGE = 50 # HARD CODED: remove far away points

class DynamicMapData:
    def __init__(self, directory):
        self.scene_id = directory.split("/")[-1]
        self.directory = Path(directory) / "pcd"
        self.pcd_files = [os.path.join(self.directory, f) for f in sorted(os.listdir(self.directory)) if f.endswith('.pcd')]

    def __len__(self):
        return len(self.pcd_files)
    
    def __getitem__(self, index_):
        res_dict = {
            'scene_id': self.scene_id,
            'timestamp': self.pcd_files[index_].split("/")[-1].split(".")[0],
        }
        pcd_ = pcdpy3.PointCloud.from_path(self.pcd_files[index_])
        pc0 = pcd_.np_data[:,:3]
        res_dict['pc'] = pc0.astype(np.float32)
        res_dict['pose'] = list(pcd_.viewpoint)
        return res_dict

def main_vis(
    data_dir: str = "/home/kin/data/00",
    voxel_map: bool = True, # output voxel-level map or raw point-level.
):
    """
    可视化动态地图数据的主函数，使用dufomap算法处理点云数据并生成静态地图
    
    参数:
        data_dir: str - 数据集目录路径，包含点云数据和位姿信息
        voxel_map: bool - 是否输出体素级地图（True）或原始点级地图（False），默认为True
    
    返回值:
        无
    
    功能流程:
        1. 初始化dufomap实例和数据加载器
        2. 遍历所有帧，进行范围过滤和点云集成
        3. 执行传播操作以更新地图
        4. 输出静态地图结果
    """
    # 加载动态地图数据集
    dataset = DynamicMapData(data_dir)

    # STEP 0: 初始化dufomap实例
    # 参数说明: 分辨率=0.1m, d_s=0.2m(静态阈值), d_p=2(传播阈值), 线程数=12
    # 这些参数与论文中的设置保持一致
    mydufo = dufomap(0.1, 0.2, 2, num_threads=12) # resolution, d_s, d_p same with paper.
    
    # 初始化累积点云数组，形状为 (0, 3)，数据类型为 float32
    # 用于存储所有帧的点云数据，后续用于提取静态点
    cloud_acc = np.zeros((0, 3), dtype=np.float32)
    
    # 遍历数据集中的所有帧，使用tqdm显示进度条
    for data_id in (pbar := tqdm(range(0, len(dataset)),ncols=100)):
        # 获取当前帧的数据，包含点云、位姿、场景ID和时间戳
        data = dataset[data_id]
        now_scene_id = data['scene_id']
        
        # 更新进度条描述信息，显示当前处理的帧ID、场景ID和时间戳
        pbar.set_description(f"id: {data_id}, scene_id: {now_scene_id}, timestamp: {data['timestamp']}")
        
        # 计算点云中每个点到传感器位置的欧氏距离
        # data['pc'][:, :3]: 点云坐标 (N, 3)
        # data['pose'][:3]: 传感器位置 (3,)
        # norm_pc0: 距离数组 (N,)，表示每个点到传感器的距离
        norm_pc0 = np.linalg.norm(data['pc'][:, :3] - data['pose'][:3], axis=1)
        
        # 创建距离范围掩码，过滤掉距离过近或过远的点
        # 保留 MIN_AXIS_RANGE < 距离 < MAX_AXIS_RANGE 的点
        # 过近的点可能是传感器自身或载体，过远的点可能噪声较大
        range_mask = (
                (norm_pc0>MIN_AXIS_RANGE) & 
                (norm_pc0<MAX_AXIS_RANGE)
        )
        
        # STEP 1: 将经过距离过滤后的点云集成到dufomap中
        # data['pc'][range_mask]: 过滤后的点云数据
        # data['pose']: 传感器位姿（包含位置和姿态）
        # cloud_transform=False: 点云已经在世界坐标系下，不需要额外的坐标变换
        mydufo.run(data['pc'][range_mask], data['pose'], cloud_transform = False)
        
        # 将当前帧的所有点云（未过滤）累积到总点云中
        # cloud_acc 形状变化: (M, 3) -> (M+N, 3)，其中 N 是当前帧点数
        cloud_acc = np.concatenate((cloud_acc, data['pc']), axis=0)
    
    # STEP 2: 执行传播操作，但不执行聚类
    # if_propagate=True: 传播八叉树中修改过的节点信息，确保地图一致性
    # if_cluster=False: 不执行聚类操作（聚类可用于进一步识别动态物体）
    mydufo.oncePropagateCluster(if_propagate=True, if_cluster=False)
    
    # STEP 3: 输出地图结果
    # cloud_acc: 累积的所有点云数据
    # voxel_map: 如果为True，输出体素级地图；如果为False，输出原始点级地图
    # 该函数会根据dufomap的分析结果，过滤掉动态物体，只保留静态场景
    mydufo.outputMap(cloud_acc, voxel_map=voxel_map)
    
    # 打印详细的计时统计信息，用于性能分析
    # 包括各个步骤（读取、集成、传播等）的耗时
    mydufo.printDetailTiming()

if __name__ == "__main__":
    start_time = time.time()
    fire.Fire(main_vis)
    print(f"Time used: {time.time() - start_time:.2f} s")