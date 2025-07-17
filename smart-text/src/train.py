# ========== 导入必要的库 ==========
# TODO: 导入时间模块，用于生成时间戳

# TODO: 导入PyTorch核心模块

# TODO: 导入TensorBoard写入器，用于记录训练过程

# TODO: 导入进度条库，用于显示训练进度

# TODO: 导入自定义模块
# - 从dataset模块导入get_dataloader函数
# - 从model模块导入InputMethodModel类
# - 导入config配置模块


# ========== 定义单轮训练函数 ==========
def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    训练一个轮次(epoch)的函数
    :param model: 神经网络模型
    :param dataloader: 数据加载器，提供批次数据
    :param loss_function: 损失函数，计算预测与真实值的差异
    :param optimizer: 优化器，更新模型参数
    :param device: 计算设备(CPU或GPU)
    :return: 当前epoch的平均损失值
    """
    # TODO: 初始化总损失为0

    # TODO: 将模型设置为训练模式(启用dropout、batch normalization等)

    # TODO: 遍历数据加载器中的每个批次，使用tqdm显示进度条
    # 提示：for inputs, targets in tqdm(dataloader, desc="训练"):
    # TODO: 将数据移动到指定设备(GPU或CPU)
    # 提示：inputs = inputs.to(device)
    # 提示：targets = targets.to(device)

    # TODO: 清零梯度，防止梯度累积
    # 提示：optimizer.zero_grad()

    # TODO: 前向传播：通过模型计算预测结果
    # 提示：outputs = model(inputs)

    # TODO: 计算损失：比较预测结果与真实标签
    # 提示：loss = loss_function(outputs, targets)

    # TODO: 反向传播：计算梯度
    # 提示：loss.backward()

    # TODO: 参数更新：根据梯度更新模型参数
    # 提示：optimizer.step()

    # TODO: 累积损失值(转换为Python数值)
    # 提示：epoch_total_loss += loss.item()

    # TODO: 返回当前epoch的平均损失
    # 提示：return epoch_total_loss / len(dataloader)


# ========== 定义主训练函数 ==========
def train():
    """主训练函数，负责整个训练流程的协调"""

    # ========== 1. 设备配置 ==========
    # TODO: 自动选择可用的计算设备：优先使用GPU，否则使用CPU
    # 提示：device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # TODO: 打印设备信息
    # 提示：print(f"设备:{device}")

    # ========== 2. 数据准备 ==========
    # TODO: 创建训练数据加载器
    # 提示：dataloader = get_dataloader()
    # TODO: 打印数据集加载完成信息

    # TODO: 加载词汇表文件，获取所有字符的列表
    # 提示：with open(config.PROCESSED_DIR / "vocab.txt", "r", encoding="utf-8") as f:
    #     # 读取每一行并去除换行符，构建词汇表列表
    #     vocab_list = [line[:-1] for line in f.readlines()]
    # TODO: 打印词表加载完成信息

    # ========== 3. 模型初始化 ==========
    # TODO: 创建输入法模型实例，词汇表大小等于字符种类数
    # 提示：model = InputMethodModel(vocab_size=len(vocab_list)).to(device)

    # ========== 4. 训练组件配置 ==========
    # TODO: 定义交叉熵损失函数，适用于多分类问题
    # 提示：loss_function = torch.nn.CrossEntropyLoss()

    # TODO: 定义Adam优化器，用于更新模型参数
    # 提示：optimizer = torch.optim.Adam(
    #     model.parameters(),           # 模型的所有可训练参数
    #     lr=config.LEARNING_RATE      # 学习率
    # )

    # ========== 5. 日志记录配置 ==========
    # TODO: 创建TensorBoard写入器，用于可视化训练过程
    # 日志目录使用当前时间戳命名
    # 提示：writer = SummaryWriter(
    #     log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")
    # )

    # ========== 6. 开始训练循环 ==========
    # TODO: 初始化最佳损失为无穷大，用于模型保存判断
    # 提示：best_loss = float("inf")

    # TODO: 遍历每个训练轮次
    # 提示：for epoch in range(1, config.EPOCHS + 1):
    # TODO: 打印当前epoch信息
    # 提示：print(f"========== Epoch {epoch} ==========")

    # TODO: 执行一个epoch的训练，获取平均损失
    # 提示：avg_loss = train_one_epoch(
    #     model,          # 模型
    #     dataloader,     # 数据加载器
    #     loss_function,  # 损失函数
    #     optimizer,      # 优化器
    #     device          # 计算设备
    # )

    # TODO: 打印当前epoch的损失值
    # 提示：print(f"Loss:{avg_loss}")

    # TODO: 将损失值记录到TensorBoard，用于可视化
    # 提示：writer.add_scalar("Loss", avg_loss, epoch)

    # ========== 7. 模型保存逻辑 ==========
    # TODO: 如果当前损失比历史最佳损失更小，则保存模型
    # 提示：if avg_loss < best_loss:
    #     best_loss = avg_loss  # 更新最佳损失
    #     # 保存模型的状态字典(参数)
    #     torch.save(model.state_dict(), config.MODELS_DIR / "model.pt")
    #     print("保存模型成功")
    # else:
    #     print("无需保存模型")


# ========== 程序入口点 ==========
# TODO: 当直接运行此脚本时执行训练
# 提示：if __name__ == "__main__":
#     train()


# ========== 编程提示 ==========
"""
📝 编程步骤建议：

1. 首先完成所有import语句
2. 实现train_one_epoch函数的循环体
3. 实现train函数的各个配置步骤
4. 实现训练循环和模型保存逻辑
5. 添加程序入口点

🎯 关键概念理解：
- 前向传播：数据通过模型得到预测结果
- 损失计算：比较预测结果与真实答案的差距
- 反向传播：计算如何调整参数来减少损失
- 参数更新：实际调整模型参数

🚀 调试技巧：
- 先用小数据集测试
- 打印中间结果的形状
- 观察损失值是否下降
- 使用TensorBoard可视化训练过程
"""
import torch
import config
import time
from tqdm import tqdm
from dataset import get_dataloader
from model import InputMethodModel
from torch.utils.tensorboard import SummaryWriter


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备:{device}")

    # 1. 创建数据加载器
    data_loader = get_dataloader()
    print("数据加载器创建完毕")

    # 2. 加载词汇表
    with open(config.PROCESSED_DIR / "vocab.txt", "r", encoding="utf-8") as f:
        vocab_list = [line[:-1] for line in f.readlines()]
    print(f"词表大小:{ len(vocab_list) }")

    # 3. 模型初始化
    model = InputMethodModel(vocab_size=len(vocab_list)).to(device)
    print("模型初始化完毕")

    # 4. 训练组件配置

    # 4.1 损失函数，定义交叉熵损失函数，用于多分类问题
    loss_function = torch.nn.CrossEntropyLoss()

    # 4.2 优化器，定义Adam优化器，学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # tensorboard write
    writer = SummaryWriter(log_dir=config.LOGS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S"))

    # 5. 开始训练
    best_loss = float("inf")
    for epoch in range(1, config.EPOCHS + 1):
        print(f"========== Epoch {epoch} ==========")

        avg_loss = train_one_epoch(model, data_loader, loss_function, optimizer, device)
        print(f"Loss:{avg_loss}")

        # 记录训练结果
        writer.add_scalar("Loss", avg_loss, epoch)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODELS_DIR / "model.pt")
            print("模型保存成功")
        else:
            print("无需保存模型")
    
    # 关闭TensorBoard writer
    writer.close()
    print(f"训练完成！TensorBoard日志已保存到: {log_dir}")
    print(f"启动TensorBoard命令: tensorboard --logdir={config.LOGS_DIR}")


def train_one_epoch(model, dataloader, loss_function, optimizer, device):
    """
    训练一个轮次(epoch)的函数
    :param model: 神经网络模型
    :param dataloader: 数据加载器，提供批次数据
    :param loss_function: 损失函数，计算预测与真实值的差异
    :param optimizer: 优化器，更新模型参数
    :param device: 计算设备(CPU或GPU)
    :return: 当前epoch的平均损失值
    """

    model.train()

    epoch_total_loss = 0

    for inputs, targets in tqdm(dataloader, desc="训练"):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # 清零梯度，防止梯度累积
        optimizer.zero_grad()

        outputs = model(inputs)

        # 计算损失，比较预测结果和真实结果之间的差异
        loss = loss_function(outputs, targets)

        # 反向传播，计算梯度
        loss.backward()

        # 更新参数
        optimizer.step()

        epoch_total_loss += loss.item()

    return epoch_total_loss / len(dataloader)


if __name__ == "__main__":
    train()
