#!/usr/bin/env python3
"""JSONL数据预处理脚本"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict

def load_jsonl_file(file_path: str) -> List[Dict]:
    """加载JSONL文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    item = json.loads(line)
                    data.append(item)
                except json.JSONDecodeError:
                    continue
    return data

def extract_conversations(data: List[Dict], is_chinese: bool = True) -> List[Dict[str, str]]:
    """从JSONL数据中提取对话"""
    conversations = []
    
    for item in data:
        if 'conversation' in item and isinstance(item['conversation'], list):
            for conv in item['conversation']:
                if 'human' in conv and 'assistant' in conv:
                    human_text = conv['human'].strip()
                    assistant_text = conv['assistant'].strip()
                    
                    # 过滤掉空文本
                    if not human_text or not assistant_text:
                        continue
                    
                    # 根据文件类型分配中英文
                    if is_chinese:
                        conversations.append({
                            'zh': human_text,
                            'en': assistant_text
                        })
                    else:
                        conversations.append({
                            'zh': assistant_text,
                            'en': human_text
                        })
    
    return conversations

def clean_and_filter_data(data: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """清理和过滤数据"""
    cleaned_data = []
    
    for item in data:
        zh_text = item['zh'].strip()
        en_text = item['en'].strip()
        
        # 过滤条件
        if (
            len(zh_text) > 0 and len(en_text) > 0 and  # 非空
            len(zh_text) < 500 and len(en_text) < 500 and  # 长度限制
            not zh_text.startswith('http') and not en_text.startswith('http')  # 过滤URL
        ):
            cleaned_data.append({
                'zh': zh_text,
                'en': en_text
            })
    
    return cleaned_data

def process_jsonl_data(
    zh_jsonl_path: str,
    en_jsonl_path: str,
    output_dir: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    处理JSONL数据集
    
    Args:
        zh_jsonl_path: 中文JSONL文件路径
        en_jsonl_path: 英文JSONL文件路径
        output_dir: 输出目录
        train_ratio: 训练集比例
        valid_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    """
    # 设置随机种子
    random.seed(seed)
    
    print(f"正在加载中文数据: {zh_jsonl_path}")
    zh_data = load_jsonl_file(zh_jsonl_path)
    print(f"中文数据加载完成，共 {len(zh_data)} 条记录")
    
    print(f"正在加载英文数据: {en_jsonl_path}")
    en_data = load_jsonl_file(en_jsonl_path)
    print(f"英文数据加载完成，共 {len(en_data)} 条记录")
    
    # 提取对话
    print("正在提取中文对话...")
    zh_conversations = extract_conversations(zh_data, is_chinese=True)
    print(f"中文对话提取完成，共 {len(zh_conversations)} 条")
    
    print("正在提取英文对话...")
    en_conversations = extract_conversations(en_data, is_chinese=False)
    print(f"英文对话提取完成，共 {len(en_conversations)} 条")
    
    # 合并数据
    all_conversations = zh_conversations + en_conversations
    print(f"合并后共 {len(all_conversations)} 条对话")
    
    # 清理和过滤数据
    print("正在清理和过滤数据...")
    cleaned_data = clean_and_filter_data(all_conversations)
    print(f"清理后共 {len(cleaned_data)} 条有效对话")
    
    # 去重
    seen = set()
    unique_data = []
    for item in cleaned_data:
        key = (item['zh'], item['en'])
        if key not in seen:
            seen.add(key)
            unique_data.append(item)
    
    print(f"去重后共 {len(unique_data)} 条唯一对话")
    
    # 打乱数据
    random.shuffle(unique_data)
    
    # 计算分割点
    total_size = len(unique_data)
    train_size = int(total_size * train_ratio)
    valid_size = int(total_size * valid_ratio)
    
    # 分割数据
    train_data = unique_data[:train_size]
    valid_data = unique_data[train_size:train_size + valid_size]
    test_data = unique_data[train_size + valid_size:]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存分割后的数据
    datasets = {
        'train': train_data,
        'val': valid_data,  # 注意这里使用val而不是valid
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
    zh_jsonl_path = project_root / "data" / "raw" / "common_zh_70k.jsonl"
    en_jsonl_path = project_root / "data" / "raw" / "common_en_70k.jsonl"
    output_dir = project_root / "data" / "processed"
    
    # 检查JSONL文件是否存在
    if not zh_jsonl_path.exists():
        print(f"错误: 找不到中文数据文件 {zh_jsonl_path}")
        return
    
    if not en_jsonl_path.exists():
        print(f"错误: 找不到英文数据文件 {en_jsonl_path}")
        return
    
    print("开始JSONL数据预处理...")
    
    # 处理数据
    datasets = process_jsonl_data(
        str(zh_jsonl_path),
        str(en_jsonl_path),
        str(output_dir),
        train_ratio=0.8,
        valid_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    # 打印统计信息
    print("\n数据分割完成:")
    print(f"训练集: {len(datasets['train'])} 条")
    print(f"验证集: {len(datasets['val'])} 条")
    print(f"测试集: {len(datasets['test'])} 条")
    print(f"总计: {sum(len(d) for d in datasets.values())} 条")
    
    # 显示一些示例数据
    print("\n示例数据:")
    for i, item in enumerate(datasets['train'][:3]):
        print(f"  {i+1}. 中文: {item['zh']}")
        print(f"     英文: {item['en']}")
    
    print("\n✅ JSONL数据预处理完成！")

if __name__ == "__main__":
    main()