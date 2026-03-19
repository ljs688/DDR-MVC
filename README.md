# DDR-MVC（精简版）

这是一个仅保留基础训练能力的 DDR-MVC 项目，用于快速跑通多视图聚类训练流程。

项目只包含：

- `.mat` 多视图数据加载
- DDR-MVC 基础模型训练
- 聚类评估指标（ACC / NMI / ARI）
- 模型权重保存

不包含实验脚本、画图脚本和批量调参脚本。

## 目录结构

```text
DDR-MVC/
├─ data/
│  └─ MSRCV1.mat                  # 示例最小数据集
├─ ddr_mvc/
│  ├─ __init__.py
│  ├─ config.py                   # 训练配置
│  ├─ data.py                     # 数据集加载与预处理
│  ├─ model.py                    # DDR-MVC 模型定义
│  ├─ losses.py                   # 损失函数
│  ├─ metrics.py                  # 聚类指标计算
│  └─ trainer.py                  # 训练与评估主流程
├─ checkpoints/                   # 保存训练权重
├─ train.py                       # 训练入口脚本
├─ requirements.txt               # 依赖列表
└─ README.md
```

## 使用方法

1. 安装依赖

```bash
pip install -r requirements.txt
```

2. 启动训练（默认读取 `data/MSRCV1.mat`）

```bash
python train.py
```

3. 常用参数示例

```bash
python train.py --dataset-name MSRCV1 --epochs 100 --batch-size 128
```

## 输出结果

- 训练日志打印在终端
- 模型保存到 `checkpoints/ddr_mvc_<dataset>.pt`
