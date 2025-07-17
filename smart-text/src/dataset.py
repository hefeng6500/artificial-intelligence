# 1.定义Dataset - 数据集类的定义
# 导入pandas库，用于数据处理
import pandas as pd
# 导入PyTorch核心模块
import torch
# 导入PyTorch数据处理相关模块
from torch.utils.data import Dataset, DataLoader
# 导入配置文件
import config


# 定义输入法数据集类，继承自PyTorch的Dataset
class InputMethodDataset(Dataset):
    def __init__(self, data_path):
        """初始化数据集
        Args:
            data_path: 数据文件路径，JSONL格式
        """
        # 数据格式示例：[{"input":[2137,9117,5436,2747,7549],"target":7784},{"input":[9117,5436,2747,7549,7784],"target":3856}]
        # 使用pandas读取JSONL文件（每行一个JSON对象）
        self.data = pd.read_json(
            data_path,           # 文件路径
            lines=True,          # 每行一个JSON对象
            orient="records"     # 记录格式
        ).to_dict(
            orient="records"     # 转换为字典列表格式
        )

    def __len__(self):
        """返回数据集大小
        Returns:
            int: 数据集中样本的数量
        """
        return len(self.data)

    def __getitem__(self, index):
        """获取指定索引的数据样本
        Args:
            index: 样本索引
        Returns:
            tuple: (输入张量, 目标张量)
        """
        # 将输入序列转换为长整型张量
        input_tensor = torch.tensor(
            self.data[index]["input"],  # 输入字符ID列表
            dtype=torch.long            # 长整型数据类型
        )
        
        # 将目标字符转换为长整型张量
        target_tensor = torch.tensor(
            self.data[index]["target"], # 目标字符ID
            dtype=torch.long            # 长整型数据类型
        )
        
        return input_tensor, target_tensor


# 2.获取DataLoader的方法 - 创建数据加载器
def get_dataloader(train=True):
    """创建数据加载器
    Args:
        train: 是否为训练集，True为训练集，False为测试集
    Returns:
        DataLoader: PyTorch数据加载器对象
    """
    # 根据train参数选择对应的数据文件路径
    data_path = config.PROCESSED_DIR / (
        "indexed_train.jsonl" if train else "indexed_test.jsonl"
    )
    
    # 创建数据集实例
    dataset = InputMethodDataset(data_path)
    
    # 创建并返回数据加载器
    return DataLoader(
        dataset,                    # 数据集对象
        batch_size=config.BATCH_SIZE,  # 批次大小
        shuffle=True                # 是否打乱数据顺序
    )


# 主程序入口，用于测试数据加载器
if __name__ == "__main__":
    # 创建训练数据加载器
    train_dataloader = get_dataloader()
    print(f"train batch个数：{len(train_dataloader)}")
    
    # 创建测试数据加载器
    test_dataloader = get_dataloader(train=False)
    print(f"test batch个数：{len(test_dataloader)}")

    # 遍历一个批次的数据，查看数据形状
    for inputs, targets in train_dataloader:
        print(inputs.shape)  # inputs.shape: [batch_size, seq_len] - 输入序列形状
        print(targets.shape)  # targets.shape: [batch_size] - 目标标签形状
        break  # 只查看第一个批次
