"""数据加载器模块"""

import os
import json
import random
from typing import List, Dict, Tuple, Optional, Iterator
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
from .tokenizer import BilingualTokenizer


class TranslationDataset(Dataset):
    """翻译数据集类"""
    
    def __init__(
        self,
        data: List[Dict[str, str]],
        tokenizer: BilingualTokenizer,
        max_length: int = 512,
        direction: str = 'zh2en'  # 'zh2en' or 'en2zh'
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        
        # 根据翻译方向设置源语言和目标语言
        if direction == 'zh2en':
            self.src_key = 'zh'
            self.tgt_key = 'en'
            self.src_tokenizer = tokenizer.zh_tokenizer
            self.tgt_tokenizer = tokenizer.en_tokenizer
        elif direction == 'en2zh':
            self.src_key = 'en'
            self.tgt_key = 'zh'
            self.src_tokenizer = tokenizer.en_tokenizer
            self.tgt_tokenizer = tokenizer.zh_tokenizer
        else:
            raise ValueError(f"Unsupported direction: {direction}")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        src_text = item[self.src_key]
        tgt_text = item[self.tgt_key]
        
        # 编码源序列和目标序列
        src_ids = self.src_tokenizer.encode(src_text, add_special_tokens=True)
        tgt_ids = self.tgt_tokenizer.encode(tgt_text, add_special_tokens=True)
        
        # 截断到最大长度
        if len(src_ids) > self.max_length:
            src_ids = src_ids[:self.max_length]
        if len(tgt_ids) > self.max_length:
            tgt_ids = tgt_ids[:self.max_length]
        
        # 创建解码器输入（去掉最后一个 token）和标签（去掉第一个 token）
        decoder_input_ids = tgt_ids[:-1] if len(tgt_ids) > 1 else [self.tgt_tokenizer.sos_token_id]
        label_ids = tgt_ids[1:] if len(tgt_ids) > 1 else [self.tgt_tokenizer.eos_token_id]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(decoder_input_ids, dtype=torch.long),
            'labels': torch.tensor(label_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }


class TranslationCollator:
    """翻译数据整理器"""
    
    def __init__(self, pad_token_id: int = 0, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # 提取各个字段
        src_ids = [item['src_ids'] for item in batch]
        tgt_ids = [item['tgt_ids'] for item in batch]
        labels = [item['labels'] for item in batch]
        src_texts = [item['src_text'] for item in batch]
        tgt_texts = [item['tgt_text'] for item in batch]
        
        # 填充序列
        src_ids_padded = pad_sequence(
            src_ids, batch_first=True, padding_value=self.pad_token_id
        )
        tgt_ids_padded = pad_sequence(
            tgt_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels_padded = pad_sequence(
            labels, batch_first=True, padding_value=self.ignore_index
        )
        
        return {
            'src_ids': src_ids_padded,
            'tgt_ids': tgt_ids_padded,
            'labels': labels_padded,
            'src_texts': src_texts,
            'tgt_texts': tgt_texts
        }


class TranslationDataLoader:
    """翻译数据加载器"""
    
    def __init__(
        self,
        data_path: str,
        tokenizer: BilingualTokenizer,
        batch_size: int = 32,
        max_length: int = 512,
        direction: str = 'zh2en',
        shuffle: bool = True,
        num_workers: int = 0
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.direction = direction
        self.shuffle = shuffle
        self.num_workers = num_workers
        
        # 加载数据
        self.data = self._load_data()
        
        # 创建数据集
        self.dataset = TranslationDataset(
            self.data, tokenizer, max_length, direction
        )
        
        # 创建整理器
        pad_token_id = (
            tokenizer.zh_tokenizer.pad_token_id if direction == 'zh2en' 
            else tokenizer.en_tokenizer.pad_token_id
        )
        self.collator = TranslationCollator(pad_token_id)
        
        # 创建数据加载器
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self.collator,
            pin_memory=torch.cuda.is_available()
        )
    
    def _load_data(self) -> List[Dict[str, str]]:
        """加载数据文件"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
        
        if self.data_path.endswith('.json'):
            return self._load_json_data()
        elif self.data_path.endswith('.jsonl'):
            return self._load_jsonl_data()
        elif self.data_path.endswith('.csv'):
            return self._load_csv_data()
        elif self.data_path.endswith('.txt'):
            return self._load_txt_data()
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
    
    def _load_json_data(self) -> List[Dict[str, str]]:
        """加载 JSON 格式数据"""
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        else:
            raise ValueError("JSON data should be a list of dictionaries")
    
    def _load_jsonl_data(self) -> List[Dict[str, str]]:
        """加载 JSONL 格式数据"""
        data = []
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        # 从conversation中提取中英文对话
                        if 'conversation' in item and isinstance(item['conversation'], list):
                            for conv in item['conversation']:
                                if 'human' in conv and 'assistant' in conv:
                                    # 根据文件名判断语言
                                    if 'zh' in self.data_path.lower():
                                        data.append({
                                            'zh': conv['human'],
                                            'en': conv['assistant']
                                        })
                                    else:
                                        data.append({
                                            'zh': conv['assistant'],
                                            'en': conv['human']
                                        })
                    except json.JSONDecodeError:
                        continue
        return data
    
    def _load_csv_data(self) -> List[Dict[str, str]]:
        """加载 CSV 格式数据"""
        df = pd.read_csv(self.data_path)
        
        # 检查列名，支持不同的列名格式
        if 'zh' in df.columns and 'en' in df.columns:
            # 标准格式
            return df[['zh', 'en']].to_dict('records')
        elif '0' in df.columns and '1' in df.columns:
            # 数字列名格式，重命名为标准格式
            df_renamed = df.rename(columns={'0': 'zh', '1': 'en'})
            return df_renamed[['zh', 'en']].to_dict('records')
        elif len(df.columns) >= 2:
            # 使用前两列作为中英文对照
            col1, col2 = df.columns[0], df.columns[1]
            df_renamed = df.rename(columns={col1: 'zh', col2: 'en'})
            return df_renamed[['zh', 'en']].to_dict('records')
        else:
            raise ValueError("CSV file should have at least 2 columns for Chinese and English text")
    
    def _load_txt_data(self) -> List[Dict[str, str]]:
        """加载文本格式数据（假设中英文用制表符分隔）"""
        data = []
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        data.append({
                            'zh': parts[0].strip(),
                            'en': parts[1].strip()
                        })
        
        return data
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        return len(self.dataloader)


def create_sample_data(num_samples: int = 1000, save_path: str = "sample_data.json"):
    """创建示例数据"""
    # 简单的中英文对照数据
    sample_pairs = [
        ("你好", "hello"),
        ("世界", "world"),
        ("我爱你", "I love you"),
        ("谢谢", "thank you"),
        ("再见", "goodbye"),
        ("早上好", "good morning"),
        ("晚上好", "good evening"),
        ("今天天气很好", "the weather is nice today"),
        ("我正在学习中文", "I am learning Chinese"),
        ("这是一本书", "this is a book"),
        ("我喜欢吃苹果", "I like eating apples"),
        ("他是我的朋友", "he is my friend"),
        ("我们去公园吧", "let's go to the park"),
        ("今天是星期一", "today is Monday"),
        ("我住在北京", "I live in Beijing"),
        ("这个电影很有趣", "this movie is interesting"),
        ("我需要帮助", "I need help"),
        ("请问现在几点了", "what time is it now"),
        ("我想喝水", "I want to drink water"),
        ("祝你生日快乐", "happy birthday to you")
    ]
    
    data = []
    for i in range(num_samples):
        zh, en = random.choice(sample_pairs)
        
        # 添加一些变化
        if random.random() < 0.3:
            zh = zh + "。"
        if random.random() < 0.3:
            en = en + "."
        
        data.append({
            'zh': zh,
            'en': en,
            'id': i
        })
    
    # 保存数据
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"创建了 {num_samples} 条示例数据，保存到 {save_path}")
    return data


def create_data_loaders(
    train_data_path: str,
    valid_data_path: str,
    test_data_path: str,
    tokenizer: BilingualTokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    direction: str = 'zh2en',
    num_workers: int = 0
) -> Tuple[TranslationDataLoader, TranslationDataLoader, TranslationDataLoader]:
    """创建训练、验证和测试数据加载器"""
    
    train_loader = TranslationDataLoader(
        train_data_path, tokenizer, batch_size, max_length, direction, 
        shuffle=True, num_workers=num_workers
    )
    
    valid_loader = TranslationDataLoader(
        valid_data_path, tokenizer, batch_size, max_length, direction,
        shuffle=False, num_workers=num_workers
    )
    
    test_loader = TranslationDataLoader(
        test_data_path, tokenizer, batch_size, max_length, direction,
        shuffle=False, num_workers=num_workers
    )
    
    return train_loader, valid_loader, test_loader


def split_data(
    data_path: str,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
    test_ratio: float = 0.1,
    output_dir: str = "./data/processed",
    seed: int = 42
):
    """分割数据集"""
    # 设置随机种子
    random.seed(seed)
    
    # 加载数据
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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


class InferenceDataLoader:
    """推理数据加载器"""
    
    def __init__(
        self,
        tokenizer: BilingualTokenizer,
        max_length: int = 512,
        direction: str = 'zh2en'
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.direction = direction
        
        if direction == 'zh2en':
            self.src_tokenizer = tokenizer.zh_tokenizer
            self.tgt_tokenizer = tokenizer.en_tokenizer
        else:
            self.src_tokenizer = tokenizer.en_tokenizer
            self.tgt_tokenizer = tokenizer.zh_tokenizer
    
    def encode_text(self, text: str) -> torch.Tensor:
        """编码单个文本"""
        ids = self.src_tokenizer.encode(text, add_special_tokens=True)
        
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        
        return torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    
    def decode_ids(self, ids: torch.Tensor) -> str:
        """解码 ID 序列"""
        if ids.dim() > 1:
            ids = ids.squeeze(0)
        
        ids_list = ids.tolist()
        return self.tgt_tokenizer.decode(ids_list, skip_special_tokens=True)


if __name__ == "__main__":
    # 创建示例数据
    sample_data = create_sample_data(100, "sample_data.json")
    
    # 分割数据
    datasets = split_data("sample_data.json", output_dir="./test_data")
    
    # 创建分词器（这里使用简化版本）
    from .tokenizer import create_tokenizer_from_data
    
    zh_texts = [item['zh'] for item in sample_data]
    en_texts = [item['en'] for item in sample_data]
    
    tokenizer = create_tokenizer_from_data(zh_texts, en_texts, min_freq=1, max_vocab_size=1000)
    
    # 创建数据加载器
    train_loader = TranslationDataLoader(
        "./test_data/train.json",
        tokenizer,
        batch_size=4,
        max_length=64,
        direction='zh2en'
    )
    
    # 测试数据加载
    for i, batch in enumerate(train_loader):
        if i >= 2:  # 只测试前两个批次
            break
        
        print(f"\n批次 {i + 1}:")
        print(f"源序列形状: {batch['src_ids'].shape}")
        print(f"目标序列形状: {batch['tgt_ids'].shape}")
        print(f"标签形状: {batch['labels'].shape}")
        print(f"源文本示例: {batch['src_texts'][0]}")
        print(f"目标文本示例: {batch['tgt_texts'][0]}")
    
    print("\n✅ 数据加载器测试通过")