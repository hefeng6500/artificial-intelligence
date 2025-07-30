#!/usr/bin/env python3
"""数据预处理脚本"""

import os
import json
import random
import pandas as pd
from pathlib import Path
from typing import List, Dict

def split_csv_data(
    csv_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    分割CSV数据集为训练集、验证集和测试集
    
    Args:
        csv_path: CSV文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    # 读取CSV文件
    print(f"正在读取数据文件: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # 检查列名并重命名
    if '0' in df.columns and '1' in df.columns:
        df = df.rename(columns={'0': 'zh', '1': 'en'})
    elif len(df.columns) >= 2:
        col1, col2 = df.columns[0], df.columns[1]
        df = df.rename(columns={col1: 'zh', col2: 'en'})
    
    # 确保有zh和en列
    if 'zh' not in df.columns or 'en' not in df.columns:
        raise ValueError("无法找到中文和英文列")
    
    # 清理数据
    df = df.dropna(subset=['zh', 'en'])  # 删除空值
    df = df[df['zh'].str.strip() != '']  # 删除空字符串
    df = df[df['en'].str.strip() != '']
    
    print(f"数据清理后共有 {len(df)} 条记录")
    
    # 转换为字典列表
    data = df[['zh', 'en']].to_dict('records')
    
    # 打乱数据
    random.shuffle(data)
    
    # 计算分割点
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    
    # 分割数据
    train_data = data[:train_size]
    valid_data = data[train_size:train_size + valid_size]
    test_data = data[train_size + valid_size:]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割后的数据
    datasets = {
        'train': train_data,
        'valid': valid_data,
        'test': test_data
    }
    
    for split_name, split_data in datasets.items():
        output_path = os.path.join(output_dir, f"{split_name}.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        
        print(f"{split_name} 数据: {len(split_data)} 条，保存到 {output_path}")
    
    return datasets

def main():
    """主函数"""
    # 设置路径
    project_root = Path(__file__).parent
    csv_path = project_root / "data" / "raw" / "damo_mt_testsets_zh2en_spoken_iwslt1617.csv"
    output_dir = project_root / "data" / "processed"
    
    # 检查CSV文件是否存在
    if not csv_path.exists():
        print(f"错误: 找不到数据文件 {csv_path}")
        return
    
    print("开始数据预处理...")
    
    # 分割数据
    datasets = split_csv_data(
        str(csv_path),
        str(output_dir),
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # 打印统计信息
    print("\n数据分割完成:")
    print(f"训练集: {len(datasets['train'])} 条")
    print(f"验证集: {len(datasets['valid'])} 条")
    print(f"测试集: {len(datasets['test'])} 条")
    print(f"总计: {sum(len(d) for d in datasets.values())} 条")
    
    # 显示一些示例数据
    print("\n示例数据:")
    for i, item in enumerate(datasets['train'][:3]):
        print(f"  {i+1}. 中文: {item['zh']}")
        print(f"     英文: {item['en']}")
    
    print("\n✅ 数据预处理完成！")

if __name__ == "__main__":
    main()