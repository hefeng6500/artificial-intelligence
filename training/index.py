# simple_math_model.py
# 从零构建一个字符级加减法模型

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file, load_file


# ===================== 1. 数据集生成 =====================
# 自定义数据集类，生成算术表达式及答案
class ArithmeticDataset(Dataset):
    def __init__(self, num_samples=10000, max_num=5):
        """
        Args:
            num_samples: 生成样本总数
            max_num: 参与运算的最大数字（0到max_num-1）
        """
        self.samples = []
        # 生成num_samples个随机算术表达式
        for _ in range(num_samples):
            a = random.randint(0, max_num - 1)  # 随机生成第一个操作数
            b = random.randint(0, max_num - 1)  # 随机生成第二个操作数
            op = random.choice(["+", "-"])  # 随机选择加减号
            result = a + b if op == "+" else a - b  # 计算结果
            # 构建输入输出格式：输入如'4 + 5 ='，输出'9'
            x = f"{a} {op} {b} ="
            y = str(result)
            self.samples.append((x, y))

        # 构建字符级 vocab
        chars = sorted(set("0123456789+-= "))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, s, maxlen):
        """将字符串编码为字符索引序列，不足长度用空格填充"""
        return [self.char2idx[c] for c in s.ljust(maxlen)]  # ljust保证固定长度

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """获取单个样本的输入输出编码"""
        x, y = self.samples[idx]
        # 输入编码为固定长度10，输出编码为固定长度5
        x_ids = self.encode(x, 10)  
        y_ids = self.encode(y, 5)  
        return torch.tensor(x_ids), torch.tensor(y_ids)


# ===================== 2. 模型定义 =====================
# 简单的数学运算模型，基于RNN架构
class TinyMathModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        """
        Args:
            vocab_size: 词汇表大小
            emb_dim: 嵌入层维度
            hidden_dim: RNN隐藏层维度
        """
        super().__init__()
        # 词嵌入层：将字符索引转为向量
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # GRU循环神经网络：处理序列数据
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        # 线性层：将隐藏状态映射到输出空间
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        """前向传播过程
        Args:
            x: 输入张量 shape [batch_size, seq_length]
        Returns:
            输出张量 shape [batch_size, seq_length, vocab_size]
        """
        x = self.embedding(x)
        out, _ = self.rnn(x)
        logits = self.linear(out)
        return logits


# ===================== 3. 训练函数 =====================
def train():
    """训练流程主函数"""
    dataset = ArithmeticDataset()
    # 创建数据加载器，每次读取64个样本
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = TinyMathModel(dataset.vocab_size)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # 交叉熵损失函数
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for xb, yb in dataloader:
            logits = model(xb)
            # 取最后5个字符的输出计算损失，对应答案部分
            loss = loss_fn(
                logits[:, -5:].reshape(-1, dataset.vocab_size), yb.reshape(-1)
            )
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: loss = {total_loss:.4f}")

    # torch.save(model.state_dict(), "math_model.pth")
    save_file(model.state_dict(), "math_model.safetensors")

    return model, dataset


# ===================== 4. 推理函数 =====================
def predict(model, dataset, expr):
    """模型推理函数
    Args:
        model: 训练好的模型
        dataset: 数据集对象（包含编码器）
        expr: 待预测的算术表达式
    Returns:
        预测结果字符串
    """
    model.eval()  # 设置为评估模式
    # 将输入表达式编码为索引
    x = dataset.encode(expr, 10)
    x_tensor = torch.tensor([x])  # 添加batch维度
    with torch.no_grad():  # 禁用梯度计算
        logits = model(x_tensor)
        # 取最后5个字符作为预测结果
        preds = torch.argmax(logits[:, -5:], dim=-1)[0]  
        # 将索引转回字符
        out = "".join([dataset.idx2char[i.item()] for i in preds])
    return out.strip()


if __name__ == "__main__":
    # 训练模型并获取测试结果
    model, dataset = train()
    # 测试几个示例
    examples = ["4 + 5 =", "1 - 3 =", "4 + 2 =", "4 - 4 ="]
    for expr in examples:
        print(f"{expr} {predict(model, dataset, expr)}")

    tensors = load_file("math_model.safetensors")
    print(tensors.keys())  # 查看所有参数名

    print(tensors["embedding.weight"].shape)  # 打印嵌入层权重形状
    print(tensors["embedding.weight"])  # 查看具体参数值
