# BERT-CLS 中文文本分类项目

项目位置：`D:\bert-text-classification`。已使用本机 GPU 完成训练，可以直接预测，无需重新安装或训练。

本项目对应 [HarderThenHarder/transformers_tasks 的 BERT-CLS 示例](https://github.com/HarderThenHarder/transformers_tasks/blob/main/text_classification/train.sh)。采用中文 BERT 编码文本，通过 `[CLS]` 的池化表示、Dropout 和线性分类层预测 8 个类别，微调整个模型。

## 直接使用

双击项目中的 `predict.bat`，输入中文评论后回车，即可看到类别和分数；输入 `q` 退出。

PowerShell 中也可以执行：

```powershell
cd D:\bert-text-classification
.\.venv\Scripts\python.exe inference.py --text "酒店房间干净，前台服务很好" "苹果又甜又脆"
```

批量预测（UTF-8 文件，每行一条文本）：

```powershell
.\.venv\Scripts\python.exe inference.py --input-file examples.txt --output reports\my_predictions.json
```

默认自动使用 GPU；添加 `--device cpu` 可在 CPU 上运行。预测从本地加载模型，无需联网。

## 已完成的结果

| 项目 | 本次结果 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 2080 Ti |
| Python / PyTorch | 3.11.0 / 2.5.1+cu121 |
| Transformers | 4.44.2 |
| 预训练模型 | google-bert/bert-base-chinese |
| 训练 / 验证样本 | 402 / 63 |
| 训练轮数 / 最佳轮次 | 20 / 第 7 轮 |
| 最佳模型验证准确率 | **90.48%（57 / 63）** |
| 宏平均 F1 | **93.14%** |
| 加权 F1 | 90.56% |
| 随机种子 | 42 |
| 训练循环耗时 | 45.09 秒（不含依赖安装、权重准备和首次模型加载） |
| PyTorch 峰值显存分配 | 2214.46 MiB（不含桌面及其他进程） |

已重新加载磁盘中的最佳模型完成独立评估，混淆矩阵与训练时最佳结果一致。3 项检查通过，覆盖原始数据及划分、错误数据校验、CPU 上的模型加载和批量/单条预测一致性。`examples.txt` 中的 8 条演示文本均预测为预期类别，但这些演示不构成独立测试集。

这里的任务是评论**主题分类**，标签来自原仓库：

| ID | 类别 | 训练条数 | 验证条数 |
| --- | --- | --- | --- |
| 0 | 电脑 | 22 | 2 |
| 1 | 水果 | 78 | 7 |
| 2 | 平板 | 67 | 12 |
| 3 | 书籍 | 29 | 3 |
| 4 | 衣服 | 70 | 16 |
| 5 | 酒店 | 56 | 16 |
| 6 | 蒙牛 | 13 | 1 |
| 7 | 洗浴 | 67 | 6 |

验证集用于选择最佳轮次，没有单独的测试集；验证数据较少且分布不均匀，尤其“蒙牛”只有 1 条。以上数值是本次示例验证结果，不能代表其他场景的泛化效果。模型的 Softmax 分数未经校准；不属于这 8 类的文本仍然会分到某一类。

## 项目文件

```text
D:\bert-text-classification\
  .venv\                         独立 GPU Python 环境
  upstream\                      原始仓库完整快照（未修改）
  data\                          原始训练集、验证集、标签映射
  models\bert-base-chinese\      本地预训练权重
  checkpoints\comment_classify\model_best\  已训练的最佳模型
  reports\                       指标、日志、逐条预测、混淆矩阵、训练曲线
  common.py                      数据校验与评估逻辑
  train.py                       GPU 混合精度训练
  inference.py                   单条、批量和交互预测
  evaluate.py                    重新加载模型评估并绘图
  test_project.py                数据与预测检查
  prepare_model.py               下载固定版本预训练模型
  setup.ps1                      重建环境
  train.ps1 / evaluate.ps1        Windows 启动脚本
  predict.bat                    双击即可预测
  examples.txt                   8 条演示输入
  requirements.txt               核心依赖版本
  requirements-lock.txt          本次运行完整依赖快照
  provenance.json                代码、模型及数据来源
```

报告文件：`reports/evaluation.json` 包含每类 precision、recall、F1；`reports/dev_predictions.csv` 包含全部 63 条验证文本及正确与否；`reports/training_curves.png` 和 `reports/confusion_matrix.png` 可直接打开查看。

## 重新评估

```powershell
cd D:\bert-text-classification
.\.venv\Scripts\python.exe evaluate.py
.\.venv\Scripts\python.exe -m unittest -v test_project
```

## 重新训练

为保留已经完成的结果，训练程序会拒绝覆盖已有模型。重新训练时指定新的目录：

```powershell
cd D:\bert-text-classification
.\.venv\Scripts\python.exe -u train.py --epochs 20 --device cuda:0 --output-dir checkpoints\run2 --report-dir reports\run2
.\.venv\Scripts\python.exe evaluate.py --model checkpoints\run2\model_best --report-dir reports\run2
.\.venv\Scripts\python.exe inference.py --model checkpoints\run2\model_best --text "这件衣服很合身"
```

默认参数与原始脚本的主要设置一致：中文 BERT、8 类、batch size 16、最大长度 128、20 轮、学习率 5e-5、warmup 比例 0.06。当前实现使用动态 padding、CUDA FP16 自动混合精度、梯度裁剪，并在每轮结束时评估，按宏平均 F1 保存最佳模型。只保存最佳推理权重，未保存优化器状态，因而不提供中途断点续训。

## 使用自己的数据

标签文件为 UTF-8，每行 `数字ID,类别名`，ID 必须从 0 连续递增且类别名唯一。训练和验证文件为 UTF-8，每行 `标签ID<TAB>文本`（中间必须是真正的制表符）。例如二分类可以设 `0,负面` 和 `1,正面`，并提供相应的情感标注数据。

```powershell
.\.venv\Scripts\python.exe train.py --train-path your_data\train.txt --dev-path your_data\dev.txt --labels your_data\label_mapping.txt --output-dir checkpoints\custom --report-dir reports\custom
```

程序按标签文件自动确定类别数，检查非法行、空文本、未定义标签、训练集中缺失的类别以及训练/验证文本交叉。建议另外保留独立测试集，用 `evaluate.py --data your_data\test.txt --model checkpoints\custom\model_best --report-dir reports\custom_test` 评估。超长文本默认截断到 128 tokens，可通过训练参数 `--max-length` 调整，最大 512；预测会读取训练时的长度设置。

## 环境重建

当前环境已安装完成。如需要重建，在项目目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File setup.ps1
```

脚本将使用独立 `.venv` 安装 CUDA 12.1 版 PyTorch，并在缺少模型权重时下载固定版本。环境安装及首次下载需要联网；之后可离线训练和预测。`requirements-lock.txt` 记录所有已安装版本，严格复建时可在准备好 CUDA PyTorch 后再安装此文件。

原仓库提交：`5464a2c16b73d1c8382f40fe48682830a87dfe65`。
预训练模型修订：`8f23c25b06e129b6c986331a13d8d025a92cf0ea`。
