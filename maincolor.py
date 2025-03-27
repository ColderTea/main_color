# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 19:49:32 2025

Author: Zhangchen Kong

E-mail: 18817453008@163.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei'] # 中文字体，正确显示负号
import matplotlib.patheffects as path_effects  # 字体描边
import seaborn as sns
from PIL import Image
from sklearn.cluster import KMeans
from tqdm import tqdm
import math
from skimage.color import rgb2lab, lab2rgb
import copy
from sklearn.metrics import pairwise_distances_argmin_min
from mpl_toolkits.mplot3d import Axes3D  # 导入 3D 绘图工具



class MainColor:
    '''
    主题色提取
    '''
    
    def __init__(self, name, mod):
        self.name = name # 名字
        self.img = None # 图片
        self.mod = mod # 图片扩展名，'jpg'或'png'
        self.RGB = None # 像素点数组
        self.LAB = None
        self.k = None # 聚类数量
        self.color_map_RGB = None # 色图
        self.color_map_LAB = None # 色图
        self.color_rate = None # 颜色比例
    
    def _load_img(self):
        '''
        1. 导入图片，并转换成数组
        '''
        if self.mod == 'jpg':
            img = Image.open(f'./figure/{self.name}/{self.name}.jpg') # 打卡图片
            RGB = np.array(img).reshape(-1, 3).astype(int) # 转换成数组
        elif self.mod == 'png':
            img = Image.open(f'./figure/{self.name}/{self.name}.png') # 打卡图片
            RGB = np.array(img)
            A = RGB[:, :, 3].reshape(-1)
            mask = (A >= 255)
            R = RGB[:, :, 0].reshape(-1, 1)[mask]
            G = RGB[:, :, 1].reshape(-1, 1)[mask]
            B = RGB[:, :, 2].reshape(-1, 1)[mask]
            RGB = np.hstack([R, G, B]).astype(int)
        self.img = img
        self.RGB = RGB
        self.LAB = rgb2lab(self.RGB / 255.0)
        return None
    
    def dropLAB_gray(self):
        '''
        2.1. 去除灰色
        '''
        LAB = copy.deepcopy(self.LAB)
        mask = np.sum((np.abs(LAB[:, [1, 2]]))**2, axis=1) > 8**2
        LAB = LAB[mask]
        self.LAB = LAB
        self.RGB = (lab2rgb(LAB) * 255).round().astype(int)
        return None
    
    def dropLAB_black(self):
        '''
        2.2. 去除黑色
        '''
        LAB = copy.deepcopy(self.LAB)
        mask = np.sum((np.abs(LAB - np.array([[0, 0, 0]])))**2, axis=1) > 32**2
        LAB = LAB[mask]
        self.LAB = LAB
        self.RGB = (lab2rgb(LAB) * 255).round().astype(int)
        return None
    
    def color_pure(self, alpha=0.75, rho_star=50):
        '''
        2.3. 颜色变换：提纯
        '''
        LAB = copy.deepcopy(self.LAB)

        c = rho_star**(1-alpha)
        rho = np.sqrt(LAB[:, 1]**2 + LAB[:, 2]**2)
        LAB[:, 1] = LAB[:, 1] * c * rho**(alpha-1)
        LAB[:, 2] = LAB[:, 2] * c * rho**(alpha-1)

        self.LAB = LAB
        self.RGB = (lab2rgb(LAB) * 255).round().astype(int)
        return None
    
    def color_light(self, alpha=0.33, rho_star=100):
        '''
        2.4. 颜色变换：提亮
        '''
        LAB = copy.deepcopy(self.LAB)
  
        c = rho_star**(1-alpha)
        rho = np.sqrt(LAB[:, 0]**2)
        LAB[:, 0] = LAB[:, 0] * c * rho**(alpha-1)
        
        self.LAB = LAB
        self.RGB = (lab2rgb(LAB) * 255).round().astype(int)
        return None


    def _kmeansLAB(self, k):
        '''
        3. kmeans聚类
        '''
        print('[RUN] kmeans聚类...')
        kmeans = KMeans(n_clusters=k, random_state=0)  # 创建KMeans模型并拟合数据
        kmeans.fit(self.LAB)
        labels = kmeans.labels_  # 获取聚类的标签（每个样本点所属的类别）
    
        # 找到每个簇中离聚类中心最近的样本点
        closest_points = []
        for i in range(k):
            cluster_points = self.LAB[labels == i]  # 获取当前簇的所有样本点
            if len(cluster_points) > 0:
                # 计算当前簇中每个样本点到聚类中心的距离
                distances = np.linalg.norm(cluster_points - kmeans.cluster_centers_[i], axis=1)
                # 找到距离最小的样本点
                closest_point = cluster_points[np.argmin(distances)]
                closest_points.append(closest_point)
            else:
                # 如果簇中没有样本点，使用聚类中心
                closest_points.append(kmeans.cluster_centers_[i])
    
        # 将聚类中心替换为样本中的点
        centroids = np.array(closest_points)
        # centroids = np.round(centroids)  # 取整
    
        # 统计每个簇的样本点个数
        cluster_counts = np.bincount(labels)
        cluster_rate = cluster_counts / np.sum(cluster_counts)  # 簇中样本点占比
    
        # 排序
        arg = np.argsort(-cluster_rate)
        cluster_rate = cluster_rate[arg]
        centroids = centroids[arg]
        cluster_counts = cluster_counts[arg]

        self.k = k
        self.color_map_LAB = centroids
        self.color_map_RGB = (lab2rgb(self.color_map_LAB) * 255).round().astype(int)  # 色图
        self.color_rate = cluster_rate  # 颜色比例
        return None
    
    def sort_color(self):
        '''
        4.0. 对颜色排序
        '''
        L = copy.deepcopy(self.color_map_LAB[:, 0])
        A = copy.deepcopy(self.color_map_LAB[:, 1])
        B = copy.deepcopy(self.color_map_LAB[:, 2])
        theta = np.arctan2(A, B)
        # theta = -L
        arg_theta = np.argsort(theta)
        self.color_map_LAB = self.color_map_LAB[arg_theta]
        self.color_map_RGB = self.color_map_RGB[arg_theta]
        self.color_rate = self.color_rate[arg_theta]
        return None
    
    def plot_pi(self, info=None):
        '''
        4.1. 绘图-饼图
        '''
        k = self.k
        name = self.name
        color_map = [tuple(self.color_map_RGB[i, :]) for i in range(len(self.color_map_RGB))]
        fig_name = f'{name}-{k}-饼图'
        if info:
            fig_name += f'({info})'
        fig_path = f'./figure/{name}/{fig_name}.png'
        labels = [(int(i[0]), int(i[1]), int(i[2])) for i in color_map]
    
        # 绘制饼图
        plt.figure(figsize=(8, 8), dpi=300)
        wedges, texts, autotexts = plt.pie(
            self.color_rate, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90, 
            colors=np.array(color_map) / 255,
            labeldistance=1.1,  # 调整标签距离圆心的距离
            textprops={'fontsize': 10, 'color': 'black', 'rotation_mode': 'anchor'}  # 设置文本属性
        )
    
        # 调整标签的旋转方向和颜色
        for text, color in zip(texts, color_map):
            x, y = text.get_position()  # 获取标签的坐标
            theta = np.arctan(y/(x+1e-8))
            text.set_rotation(theta * 180 / np.pi)  # 将弧度转换为角度并设置旋转
            text.set_color(np.array(color) / 255)  # 设置标签颜色为对应的 color_map 颜色
            
            # 设置字体外边框
            text.set_path_effects([
                path_effects.withStroke(linewidth=0.1, foreground='k')  # 字体描边，宽度为 2
            ])
            
            # 设置字体大小和粗细
            text.set_fontsize(12)  # 设置字体大小为 12
            text.set_fontweight('bold')  # 设置字体粗细为粗体
    
        plt.title(fig_name)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.show()
        print('[Save] 饼图已保存！')
        print(f"[Info] '{fig_path}' \n")
        return None
    
    def plot_blocks(self, info=None):
        '''
        4.2. 绘图-色块图
        '''
        k = self.k
        color_map = [tuple(self.color_map_RGB[i, :]) for i in range(len(self.color_map_RGB))]

        name = self.name
        fig_name = f'{name}-{k}-色卡'
        if info:
            fig_name += f'({info})'
        fig_path = f'./figure/{name}/{fig_name}.png'
        line_num = math.ceil(len(color_map)/8)
        fig, axs = plt.subplots(line_num, 8, figsize=(8*1.9, line_num*1.8+0.5), dpi=300)
        
        for i, color in enumerate(color_map):
            color = tuple(int(num) for num in color)
            img = Image.new('RGB', (1, 1), color)
            if line_num == 1:
                ax = axs[i]
            else:
                ax = axs[int(i//8), int(i%8)]
            ax.set_title(color)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(img)
        plt.suptitle(fig_name, fontsize=16)  # 全局标题
        plt.savefig(fig_path, bbox_inches='tight')
        plt.show()
        print('[Save] 色卡已保存！')
        print(f"[Info] '{fig_path}' \n")
        return None
    
    def plot_RGB(self, title, info=None):
        '''
        绘图：把样本点绘制到3D散点图
        '''
        if info:
            title += f'({info})'
        print('[RUN] 绘制RGB散点图...')
        X = copy.deepcopy(self.RGB)
        # 创建 3D 图形对象
        fig = plt.figure(figsize=(10,8),dpi=300)  # 创建一个图形
        ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 子图
        
        # 提取 x, y, z 坐标
        x = X[:, 0]  # 第一列是 x 坐标
        y = X[:, 1]  # 第二列是 y 坐标
        z = X[:, 2]  # 第三列是 z 坐标
        
        # 将 RGB 值归一化到 [0, 1] 范围
        colors = copy.deepcopy(self.RGB) / 255.0
        
        # 绘制散点图
        ax.scatter(x, y, z, c=colors, marker='.', s=0.1, alpha=0.2)  # c 是颜色，s 是点的大小
        
        # 设置坐标轴标签
        ax.set_xlabel('R')
        ax.set_ylabel('G')
        ax.set_zlabel('B')
        
        # 设置图片标题
        ax.set_title(title, fontsize=14)
        
        # 保存图片
        plt.savefig(f'./figure/{self.name}/{title}.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件，dpi 是分辨率
        print(f'[SAVE] ./figure/{self.name}/{title}.png')
        
        # 显示图形
        plt.show()
        
        return None
    
    def plot_LAB(self, title, info=None):
        '''
        绘图：把样本点绘制到3D散点图
        '''
        if info:
            title += f'({info})'
        print('[RUN] 绘制LAB散点图...')
        X = copy.deepcopy(self.LAB)
        # 创建 3D 图形对象
        fig = plt.figure(figsize=(10,8),dpi=300)  # 创建一个图形
        ax = fig.add_subplot(111, projection='3d')  # 添加一个 3D 子图
        
        L = X[:, 0]  # 第一列是 x 坐标
        A = X[:, 1]  # 第二列是 y 坐标
        B = X[:, 2]  # 第三列是 z 坐标
        
        # 将 RGB 值归一化到 [0, 1] 范围
        colors = copy.deepcopy(self.RGB) / 255.0
        
        # 绘制散点图
        ax.scatter(A, B, L, c=colors, marker='.', s=0.1, alpha=0.2)  # c 是颜色，s 是点的大小
        
        
        # 设置图片标题
        ax.set_title(title, fontsize=14)
        
        # 保存图片
        plt.savefig(f'./figure/{self.name}/{title}.png', dpi=300, bbox_inches='tight')  # 保存为 PNG 文件，dpi 是分辨率
        print(f'[SAVE] ./figure/{self.name}/{title}.png')

        # 显示图形
        plt.show()
        
        return None
        


if __name__ == '__main__':
    
    # 图片名称，不加扩展名
    name = '春日'
    
    # 色卡颜色数量的列表
    k_list = [8, 16, 24, 32]
    
    mymaincolor = MainColor(name, mod='png') # 创建类。
    mymaincolor._load_img() # 导入图片，转换数组
    
    mymaincolor.plot_RGB(title='RGB散点图(原图)') # 绘制散点图
    mymaincolor.plot_LAB(title='CIELAB散点图(原图)') # 绘制散点图

    print(f'像素点数量：{len(mymaincolor.LAB)}')
    mymaincolor.dropLAB_gray() # 去除灰色
    print(f'像素点数量：{len(mymaincolor.LAB)}')
    mymaincolor.dropLAB_black() # 去除黑色
    print(f'像素点数量：{len(mymaincolor.LAB)}')
    
    
    # 挑选一个色调，或者自己修改alpha和rho_star
    
    # info = '原色调'
    # mymaincolor.color_pure(alpha=1.0, rho_star=50)
    # mymaincolor.color_light(alpha=1.0, rho_star=100)

    # info = '马卡龙色调'
    # mymaincolor.color_pure(alpha=0.50, rho_star=40)
    # mymaincolor.color_light(alpha=0.33, rho_star=105)

    info = '多巴胺色调'
    mymaincolor.color_pure(alpha=0.50, rho_star=80)
    mymaincolor.color_light(alpha=0.33, rho_star=90)
    
    # info = '中纯色调'
    # mymaincolor.color_pure(alpha=0.50, rho_star=80)
    # mymaincolor.color_light(alpha=0.70, rho_star=70)
    
    # info = '暗纯色调'
    # mymaincolor.color_pure(alpha=0.50, rho_star=80)
    # mymaincolor.color_light(alpha=0.33, rho_star=40)
    
    # info = '高亮灰色调'
    # mymaincolor.color_pure(alpha=0.30, rho_star=10)
    # mymaincolor.color_light(alpha=0.33, rho_star=105)
    
    # info = '亮灰色调'
    # mymaincolor.color_pure(alpha=0.30, rho_star=20)
    # mymaincolor.color_light(alpha=0.33, rho_star=90)
    
    # info = '中灰色调'
    # mymaincolor.color_pure(alpha=0.30, rho_star=20)
    # mymaincolor.color_light(alpha=0.70, rho_star=70)
    
    # info = '暗灰色调'
    # mymaincolor.color_pure(alpha=0.30, rho_star=20)
    # mymaincolor.color_light(alpha=0.33, rho_star=40)

    # 绘制调整色调之后的散点图
    mymaincolor.plot_RGB(title='RGB散点图(过滤)', info=info)
    mymaincolor.plot_LAB(title='CIELAB散点图(过滤)', info=info)
    

    # 生成色卡
    for k in k_list:
        mymaincolor._kmeansLAB(k) # 聚类
        mymaincolor.sort_color() # 排序
        mymaincolor.plot_pi(info=info) # 绘制饼图
        mymaincolor.plot_blocks(info=info) # 绘制色卡
        
