# 急诊双结局机器学习比较

本项目比较急诊不良结局和30天非计划急诊再就诊的预测模型。实验使用公开竞赛提供的合成就诊数据。主要评价为两任务ROC-AUC、Macro AUC、灵敏度、特异度和混淆计数。

## 数据

从 Kaggle 的 AI in Emergency Medicine – Turkiye 竞赛页面获取数据，将解压后的文件放入 `data/raw/`。需要 `train.csv`、`test.csv`、`supplemental_data.csv`、`sample_submission.csv`、`hospital_reference.csv` 和 `icd10_reference.csv`。数据及训练输出通过 `.gitignore` 排除。

## 运行环境

原实验运行于 Linux、Python 3.12 和 CPU 环境。`runtime_versions.json` 与 `requirements.txt` 记录原实验软件版本，`torch` 使用 CPU 构建。完整训练包含共同五折、多个神经网络组件组合和内层视角模型。根据运行平台安装对应的 CPU 版本依赖。

在项目根目录设置 `ML_PROJECT_ROOT` 为当前目录，再按顺序执行 `notebooks/` 中的 01、06、09、11、13、15。执行完 06 后先运行 `python scripts/initialize_submission_record.py`，为尚未上传的提交文件建立状态记录，再继续后续 Notebook。示例中的 Notebook 可通过 JupyterLab 或 `jupyter nbconvert --execute` 运行。

```bash
export ML_PROJECT_ROOT="$(pwd)"
jupyter lab
```

