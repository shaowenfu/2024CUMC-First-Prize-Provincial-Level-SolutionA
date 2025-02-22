# 螺线运动数学建模

[English](./README.md) | [中文](./README_zh.md)

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.20+-green.svg)
![SciPy](https://img.shields.io/badge/SciPy-1.7+-orange.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.4+-red.svg)

> 🏆 2024年全国大学生数学建模竞赛省级一等奖

本项目提供了一个基于阿基米德螺线的运动分析数学解决方案，主要研究位置计算、速度分析和轨迹优化。

## 📋 目录

- [问题概述](#-问题概述)
- [项目结构](#-项目结构)
- [解决方案](#-解决方案)
- [结果可视化](#-结果可视化)
- [安装与使用](#-安装与使用)
- [技术细节](#-技术细节)
- [贡献者](#-贡献者)

## 🎯 问题概述

本问题研究了阿基米德螺线上连接点的运动，具有以下特征：
- 螺距(p) = 55厘米
- 初始位置在第16圈
- 相邻点之间固定距离（第一段286厘米，其余165厘米）
- 领头点保持恒定速度（1米/秒）

## 📁 项目结构

```
.
├── codes/
│   ├── calculate_head_p.py    # 计算领头把手位置
│   ├── calculate_next_p.py    # 计算后续点位置
│   ├── calculate_v.py         # 计算速度
│   └── circle_equation.py     # 处理圆形轨迹计算
├── Results/
│   ├── problem1_animation.gif # 位置分析可视化
│   └── problem2_animation.gif # 速度分析可视化
├── competition essay/
│   └── 我的论文.pdf           # 竞赛论文
├── README.md                  # 英文文档
└── README_zh.md              # 中文文档
```

## 💡 解决方案

### 问题一：位置分析
开发了追踪螺线上所有点位置的数学模型：
- 使用弧长积分计算领头点位置
- 应用几何约束确定后续点位置
- 实现数值方法求解隐式方程

![问题一动画](./Results/problem1_animation.gif)

### 问题二：速度分析
分析相邻点之间的速度关系：
- 基于几何关系推导速度传递方程
- 实现速度比的数值解
- 分析螺线上的速度变化

![问题二动画](./Results/problem2_animation.gif)

### 问题三：轨迹优化
考虑以下因素优化运动路径：
- 最小转弯半径约束
- 平滑过渡要求
- 路径长度优化
- 关键点处的曲率分析

### 问题四：稳定性分析
研究系统的稳定性：
- 分析扰动的影响
- 研究系统对干扰的响应
- 评估不同配置的稳定性条件

### 问题五：模型扩展
扩展模型以考虑：
- 变速场景
- 不同螺线配置
- 替代几何约束
- 实际实施考虑

## 🛠 安装与使用

```bash
# 克隆仓库
git clone https://github.com/yourusername/spiral-motion-modeling.git

# 安装所需包
pip install numpy scipy matplotlib

# 运行位置分析
python codes/calculate_head_p.py

# 运行速度分析
python codes/calculate_v.py
```

## 🔧 技术细节

- **编程语言**: Python 3.8+
- **核心库**:
  - NumPy: 数值计算
  - SciPy: 积分与优化
  - Matplotlib: 可视化与动画
- **数学工具**:
  - 弧长积分
  - 隐函数求解
  - 几何优化
  - 微分方程
  - 数值方法

## 👥 贡献者

- 四川大学团队成员
- https://github.com/Gixyuan

---

[English](./README.md) | [查看论文](./competition%20essay/我的论文.pdf)