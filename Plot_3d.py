from Hyperparameters import *
from Hyperparameters import N_step, test_step
import numpy as np
import torch
import matplotlib.pyplot as plt
import importlib
import utils
importlib.reload(utils)


def calculate_outer_circle(tensor):
    # Find minimum and maximum coordinates along each axis
    min_coords, _ = torch.min(tensor, dim=0)
    max_coords, _ = torch.max(tensor, dim=0)

    # Calculate the center of the minimum bounding cube
    center = (min_coords + max_coords) / 2.0

    # Calculate the half length of the minimum bounding cube
    half_length = torch.sqrt(torch.sum((max_coords - min_coords) ** 2.0)) / 2

    # The radius of the circumscribing circle is half of the cube's half length
    radius = half_length

    return center, radius


real_x = torch.load('./Saved_data/real_x.pth')
key_pre_x = torch.load('./Saved_data/key_pre_x.pth')
real_t = torch.load('./Saved_data/real_t.pth')
fig_path = './Manuscript/Figure/Plot_3d.eps'
real_t = real_t.to('cpu').detach()
real_x = real_x.to('cpu').detach()
key_pre_x = key_pre_x.to('cpu').detach()
center, radius = calculate_outer_circle(key_pre_x)
print(center)
# 提取x、y、z坐标
x = key_pre_x[:, 0]
y = key_pre_x[:, 1]
z = key_pre_x[:, 2]

# 创建3D图
# plt.style.use('dark_background')
fig = plt.figure(dpi=300)
ax = fig.add_subplot(111, projection='3d')

# 绘制散点图
# ax.plot(x, y, z, color='#FA7F6F', alpha=0.4)
u = torch.linspace(0, 2 * torch.pi, 30)
v = torch.linspace(0, torch.pi, 30)
x = radius * torch.outer(torch.cos(u), torch.sin(v)) + center[0]
y = radius * torch.outer(torch.sin(u), torch.sin(v)) + center[1]
z = radius * torch.outer(torch.ones(u.shape),
                         torch.cos(v)) + center[2]


ax.plot_surface(x, y, z, color="#002c53",
                linestyle='dashed', alpha=0.05)

for index in range(427, 428):
    pre_x_of = torch.load('./Saved_data/key_pre_x_'+str(index)+'.pth',
                          map_location=device).cpu().detach()
    plt.plot(pre_x_of[:, 0], pre_x_of[:, 1], pre_x_of[:, 2],
             color='#ffa510', alpha=0.99, dashes=[2, 2], linewidth=1)
pre_x = torch.load('./Saved_data/key_pre_x.pth',
                   map_location=device).cpu().detach()
plt.plot(pre_x[:, 0], pre_x[:, 1], pre_x[:, 2],
         color='#3b6291', alpha=0.8, dashes=[3, 2], linewidth=1)

start_point = x0.cpu().detach()
ax.scatter(start_point[0], start_point[1],
           start_point[2], s=40, color='red', label='Start Point')
# 设置坐标轴标签
# ax.set_xlabel('X', fontsize=12)  # 设置X轴标签和字体大小
ax.set_xticks([-30, 0, 30])  # 设置z轴刻度
ax.set_xticklabels(['', '', ''])
# ax.set_ylabel('Y', fontsize=12)  # 设置Y轴标签和字体大小
ax.set_yticks([-30, 0, 30])  # 设置z轴刻度
ax.set_yticklabels(['', '', ''])
# ax.set_zlabel('Z', fontsize=12)  # 设置Z轴标签和字体大小
ax.set_zticks([0, 20, 40, 60])  # 设置z轴刻度
ax.set_zticklabels(['', '', '', ''])
ax.view_init(elev=50, azim=150)
# plt.savefig('./Manuscript/Figure/Plot_3d_2.png',
#             format='png', bbox_inches='tight')


plt.show()
