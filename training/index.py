# simple_math_model.py
# 从零构建一个字符级加减法模型

import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset, DataLoader
from safetensors.torch import save_file
from safetensors.torch import load_file


# ===================== 1. 数据集生成 =====================
class ArithmeticDataset(Dataset):
    def __init__(self, num_samples=10000, max_num=5):
        self.samples = []
        for _ in range(num_samples):
            a = random.randint(0, max_num - 1)
            b = random.randint(0, max_num - 1)
            op = random.choice(["+", "-"])
            result = a + b if op == "+" else a - b
            x = f"{a} {op} {b} ="
            y = str(result)
            self.samples.append((x, y))

        # 构建字符级 vocab
        chars = sorted(set("0123456789+-= "))
        self.char2idx = {c: i for i, c in enumerate(chars)}
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    def encode(self, s, maxlen):
        return [self.char2idx[c] for c in s.ljust(maxlen)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        x_ids = self.encode(x, 10)  # 固定长度输入
        y_ids = self.encode(y, 5)  # 固定长度输出
        return torch.tensor(x_ids), torch.tensor(y_ids)


# ===================== 2. 模型定义 =====================
class TinyMathModel(nn.Module):
    def __init__(self, vocab_size, emb_dim=64, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.rnn(x)
        logits = self.linear(out)
        return logits


# ===================== 3. 训练函数 =====================
def train():
    dataset = ArithmeticDataset()
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = TinyMathModel(dataset.vocab_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        total_loss = 0
        for xb, yb in dataloader:
            logits = model(xb)
            # 只取输出最后 5 位进行监督
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
    model.eval()
    x = dataset.encode(expr, 10)
    x_tensor = torch.tensor([x])
    with torch.no_grad():
        logits = model(x_tensor)
        preds = torch.argmax(logits[:, -5:], dim=-1)[0]  # 取输出
        out = "".join([dataset.idx2char[i.item()] for i in preds])
    return out.strip()


if __name__ == "__main__":
    model, dataset = train()
    # 推理测试
    examples = ["4 + 5 =", "1 - 3 =", "4 + 2 =", "4 - 4 ="]
    for expr in examples:
        print(f"{expr} {predict(model, dataset, expr)}")

tensors = load_file("math_model.safetensors")
print(tensors.keys())  # 查看所有参数名

print(tensors["embedding.weight"].shape)
print(tensors["embedding.weight"])  # 具体值
