"""分词器模块"""

import re
import json
import pickle
from typing import List, Dict, Tuple, Optional, Union
from collections import Counter, defaultdict
import jieba
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import torch

# 下载必要的 NLTK 数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class BaseTokenizer:
    """基础分词器类"""
    
    def __init__(self):
        self.vocab = {}
        self.idx2token = {}
        self.special_tokens = {
            'PAD': '<pad>',
            'SOS': '<sos>',
            'EOS': '<eos>',
            'UNK': '<unk>'
        }
        self.pad_token_id = 0
        self.sos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        
    def build_vocab(self, texts: List[str], min_freq: int = 2, max_vocab_size: int = 30000):
        """构建词汇表"""
        # 统计词频
        word_counts = Counter()
        for text in texts:
            tokens = self.tokenize(text)
            word_counts.update(tokens)
        
        # 添加特殊标记
        self.vocab = {token: idx for idx, token in enumerate(self.special_tokens.values())}
        self.idx2token = {idx: token for token, idx in self.vocab.items()}
        
        # 按频率排序并添加到词汇表
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        
        for word, freq in sorted_words:
            if freq >= min_freq and len(self.vocab) < max_vocab_size:
                if word not in self.vocab:
                    idx = len(self.vocab)
                    self.vocab[word] = idx
                    self.idx2token[idx] = word
        
        print(f"词汇表大小: {len(self.vocab)}")
        
    def tokenize(self, text: str) -> List[str]:
        """分词（需要子类实现）"""
        raise NotImplementedError
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """编码文本为 ID 序列"""
        tokens = self.tokenize(text)
        
        if add_special_tokens:
            tokens = [self.special_tokens['SOS']] + tokens + [self.special_tokens['EOS']]
        
        return [self.vocab.get(token, self.unk_token_id) for token in tokens]
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """解码 ID 序列为文本"""
        tokens = []
        for token_id in token_ids:
            token = self.idx2token.get(token_id, self.special_tokens['UNK'])
            
            if skip_special_tokens and token in self.special_tokens.values():
                continue
                
            tokens.append(token)
        
        return self.detokenize(tokens)
    
    def detokenize(self, tokens: List[str]) -> str:
        """将 token 列表转换为文本（需要子类实现）"""
        raise NotImplementedError
    
    def save(self, filepath: str):
        """保存分词器"""
        data = {
            'vocab': self.vocab,
            'idx2token': self.idx2token,
            'special_tokens': self.special_tokens
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载分词器"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.idx2token = {int(k): v for k, v in data['idx2token'].items()}
        self.special_tokens = data['special_tokens']
    
    def __len__(self):
        return len(self.vocab)


class ChineseTokenizer(BaseTokenizer):
    """中文分词器"""
    
    def __init__(self, use_hmm: bool = True, cut_all: bool = False):
        super().__init__()
        self.use_hmm = use_hmm
        self.cut_all = cut_all
        
        # 中文标点符号
        self.chinese_punctuation = '！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〱〲〳〴〵〶〷〸〹〺〻〼〽〾〿'
        
    def tokenize(self, text: str) -> List[str]:
        """中文分词"""
        # 预处理
        text = self._preprocess(text)
        
        # 使用 jieba 分词
        tokens = list(jieba.cut(text, cut_all=self.cut_all, HMM=self.use_hmm))
        
        # 后处理
        tokens = self._postprocess(tokens)
        
        return tokens
    
    def _preprocess(self, text: str) -> str:
        """预处理文本"""
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 标准化标点符号
        text = text.replace('，', ',')
        text = text.replace('。', '.')
        text = text.replace('！', '!')
        text = text.replace('？', '?')
        text = text.replace('；', ';')
        text = text.replace('：', ':')
        
        return text
    
    def _postprocess(self, tokens: List[str]) -> List[str]:
        """后处理 token 列表"""
        processed_tokens = []
        
        for token in tokens:
            token = token.strip()
            if token and token != ' ':
                processed_tokens.append(token)
        
        return processed_tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """将中文 token 列表转换为文本"""
        text = ''.join(tokens)
        
        # 在中英文之间添加空格
        text = re.sub(r'([\u4e00-\u9fff])([a-zA-Z])', r'\1 \2', text)
        text = re.sub(r'([a-zA-Z])([\u4e00-\u9fff])', r'\1 \2', text)
        
        return text


class EnglishTokenizer(BaseTokenizer):
    """英文分词器"""
    
    def __init__(self, lowercase: bool = True, remove_punctuation: bool = False):
        super().__init__()
        self.lowercase = lowercase
        self.remove_punctuation = remove_punctuation
        
        # 英文停用词
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            self.stop_words = set()
    
    def tokenize(self, text: str) -> List[str]:
        """英文分词"""
        # 预处理
        text = self._preprocess(text)
        
        # 使用 NLTK 分词
        try:
            tokens = word_tokenize(text)
        except:
            # 如果 NLTK 不可用，使用简单的空格分词
            tokens = text.split()
        
        # 后处理
        tokens = self._postprocess(tokens)
        
        return tokens
    
    def _preprocess(self, text: str) -> str:
        """预处理文本"""
        # 转换为小写
        if self.lowercase:
            text = text.lower()
        
        # 去除多余空格
        text = re.sub(r'\s+', ' ', text.strip())
        
        # 处理缩写
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"n't", " not", text)
        text = re.sub(r"'re", " are", text)
        text = re.sub(r"'ve", " have", text)
        text = re.sub(r"'ll", " will", text)
        text = re.sub(r"'d", " would", text)
        text = re.sub(r"'m", " am", text)
        
        return text
    
    def _postprocess(self, tokens: List[str]) -> List[str]:
        """后处理 token 列表"""
        processed_tokens = []
        
        for token in tokens:
            token = token.strip()
            
            if not token:
                continue
            
            # 移除标点符号（可选）
            if self.remove_punctuation:
                token = re.sub(r'[^\w\s]', '', token)
                if not token:
                    continue
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def detokenize(self, tokens: List[str]) -> str:
        """将英文 token 列表转换为文本"""
        text = ' '.join(tokens)
        
        # 处理标点符号
        text = re.sub(r' +', ' ', text)
        text = re.sub(r' ([.!?,:;])', r'\1', text)
        text = re.sub(r'\( ', '(', text)
        text = re.sub(r' \)', ')', text)
        text = re.sub(r'\[ ', '[', text)
        text = re.sub(r' \]', ']', text)
        
        return text


class BilingualTokenizer:
    """双语分词器"""
    
    def __init__(self):
        self.zh_tokenizer = ChineseTokenizer()
        self.en_tokenizer = EnglishTokenizer()
        
    def build_vocab(
        self, 
        zh_texts: List[str], 
        en_texts: List[str], 
        min_freq: int = 2, 
        max_vocab_size: int = 30000
    ):
        """构建双语词汇表"""
        print("构建中文词汇表...")
        self.zh_tokenizer.build_vocab(zh_texts, min_freq, max_vocab_size)
        
        print("构建英文词汇表...")
        self.en_tokenizer.build_vocab(en_texts, min_freq, max_vocab_size)
    
    def encode_pair(
        self, 
        zh_text: str, 
        en_text: str, 
        max_length: int = 512
    ) -> Tuple[List[int], List[int]]:
        """编码中英文对"""
        zh_ids = self.zh_tokenizer.encode(zh_text)
        en_ids = self.en_tokenizer.encode(en_text)
        
        # 截断或填充到指定长度
        zh_ids = self._pad_or_truncate(zh_ids, max_length, self.zh_tokenizer.pad_token_id)
        en_ids = self._pad_or_truncate(en_ids, max_length, self.en_tokenizer.pad_token_id)
        
        return zh_ids, en_ids
    
    def decode_pair(self, zh_ids: List[int], en_ids: List[int]) -> Tuple[str, str]:
        """解码中英文对"""
        zh_text = self.zh_tokenizer.decode(zh_ids)
        en_text = self.en_tokenizer.decode(en_ids)
        
        return zh_text, en_text
    
    def _pad_or_truncate(self, ids: List[int], max_length: int, pad_id: int) -> List[int]:
        """填充或截断序列"""
        if len(ids) > max_length:
            return ids[:max_length]
        else:
            return ids + [pad_id] * (max_length - len(ids))
    
    def save(self, zh_path: str, en_path: str):
        """保存双语分词器"""
        self.zh_tokenizer.save(zh_path)
        self.en_tokenizer.save(en_path)
    
    def load(self, zh_path: str, en_path: str):
        """加载双语分词器"""
        self.zh_tokenizer.load(zh_path)
        self.en_tokenizer.load(en_path)
    
    @property
    def zh_vocab_size(self) -> int:
        return len(self.zh_tokenizer)
    
    @property
    def en_vocab_size(self) -> int:
        return len(self.en_tokenizer)


def create_tokenizer_from_data(
    zh_texts: List[str],
    en_texts: List[str],
    min_freq: int = 2,
    max_vocab_size: int = 30000
) -> BilingualTokenizer:
    """从数据创建双语分词器"""
    tokenizer = BilingualTokenizer()
    tokenizer.build_vocab(zh_texts, en_texts, min_freq, max_vocab_size)
    return tokenizer


def batch_encode(
    tokenizer: Union[ChineseTokenizer, EnglishTokenizer],
    texts: List[str],
    max_length: int = 512,
    padding: bool = True
) -> torch.Tensor:
    """批量编码文本"""
    encoded_texts = []
    
    for text in texts:
        ids = tokenizer.encode(text)
        
        if len(ids) > max_length:
            ids = ids[:max_length]
        elif padding and len(ids) < max_length:
            ids = ids + [tokenizer.pad_token_id] * (max_length - len(ids))
        
        encoded_texts.append(ids)
    
    return torch.tensor(encoded_texts, dtype=torch.long)


if __name__ == "__main__":
    # 测试中文分词器
    zh_tokenizer = ChineseTokenizer()
    zh_text = "你好，世界！这是一个测试句子。"
    zh_tokens = zh_tokenizer.tokenize(zh_text)
    print(f"中文分词结果: {zh_tokens}")
    
    # 测试英文分词器
    en_tokenizer = EnglishTokenizer()
    en_text = "Hello, world! This is a test sentence."
    en_tokens = en_tokenizer.tokenize(en_text)
    print(f"英文分词结果: {en_tokens}")
    
    # 测试双语分词器
    zh_texts = ["你好世界", "这是测试", "机器翻译"]
    en_texts = ["hello world", "this is test", "machine translation"]
    
    bilingual_tokenizer = create_tokenizer_from_data(zh_texts, en_texts, min_freq=1, max_vocab_size=1000)
    
    zh_ids, en_ids = bilingual_tokenizer.encode_pair("你好世界", "hello world")
    print(f"编码结果 - 中文: {zh_ids[:10]}")
    print(f"编码结果 - 英文: {en_ids[:10]}")
    
    zh_decoded, en_decoded = bilingual_tokenizer.decode_pair(zh_ids, en_ids)
    print(f"解码结果 - 中文: {zh_decoded}")
    print(f"解码结果 - 英文: {en_decoded}")
    
    print(f"中文词汇表大小: {bilingual_tokenizer.zh_vocab_size}")
    print(f"英文词汇表大小: {bilingual_tokenizer.en_vocab_size}")
    
    print("✅ 分词器测试通过")