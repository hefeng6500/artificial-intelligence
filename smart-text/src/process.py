"""
数据预处理脚本：将原始对话数据转换为模型可训练的索引化序列

核心功能：
1. 读取原始JSONL格式的对话数据
2. 提取对话中的句子并划分训练集/测试集
3. 基于训练集构建词汇表（word2index映射）
4. 将句子转换为索引序列，生成输入-目标对（滑动窗口）
5. 保存处理后的训练集和测试集（JSONL格式）

输入：
- 原始数据路径：config.RAW_DATA_DIR / "synthesized_.jsonl"（JSONL格式，含dialog字段）

输出：
- 处理后数据路径：config.PROCESSED_DIR
  - indexed_train.jsonl：训练集（每行一个样本）
  - indexed_test.jsonl：测试集（单个JSON数组）
- 词汇表通过word2index映射隐含在数据中

依赖：
- jieba：中文分词工具
- pandas：数据读取和保存
- sklearn：数据集划分
- config：配置文件（定义数据路径和序列长度等参数）
"""

import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
import config
from tqdm import tqdm


def build_dataset(sentences, word2index):
    """
    构建数据集
    :param sentences: 句子列表，原始距离列表['我爱自然语言','我不爱自然语言']
    :param word2index: 单词到索引的映射，{word:index}
    :return: 数据集，[{input:[1,2,3,4,5],target:6},{input:[2,3,4,5,6],target:7}]
    """
    dataset = []
    for sentence in sentences:
        tokens = [word2index.get(word, 0) for word in jieba.lcut(sentence)]
        for i in range(0, len(tokens) - config.SEQ_LEN):
            input_seq = tokens[i : i + config.SEQ_LEN]
            target = tokens[i + config.SEQ_LEN]
            dataset.append({"input": input_seq, "target": target})
    return dataset


def process():
    print("开始处理数据...")

    # 1. 读取原始JSONL格式的对话数据
    df = pd.read_json(
        config.RAW_DATA_DIR / "synthesized_.jsonl", lines=True, orient="records"
    ).sample(
        frac=0.1
    )  # 采样 10% 数据，可根据需要调整（如全量处理可删除此句）

    # 2. 提取对话中的句子并划分训练集/测试集
    sentences = []
    for dialog in df["dialog"]:
        for sentence in dialog:
            sentences.append(sentence.split("：")[1])

    train_sentences, test_sentences = train_test_split(sentences, test_size=0.2)
    print(f"训练集大小:{ len(train_sentences) }")
    print(f"测试集大小:{ len(test_sentences) }")

    # 3. 基于训练集构建词汇表（word2index映射）
    vocab_set = set()
    for sentence in tqdm(train_sentences, desc="构建词汇表"):
        for word in jieba.lcut(sentence):
            vocab_set.add(word)

    # 使用 get() 方法可以给未登录词设置统一的索引，这对模型训练来说很关键。
    vocab_list = ["<unk>"] + list(vocab_set)
    print(f"词表大小:{ len(vocab_list) }")

    # 保存词表
    with open(config.PROCESSED_DIR / "vocab.txt", "w", encoding="utf-8") as f:
        for word in vocab_list:
            f.write(word + "\n")

    word2index = {word: index for index, word in enumerate(vocab_list)}

    # 4. 将句子转换为索引序列，生成输入-目标对（滑动窗口）
    train_dataset = build_dataset(train_sentences, word2index)
    test_dataset = build_dataset(test_sentences, word2index)

    print(f"训练集大小:{ len(train_dataset) }")
    print(f"测试集大小:{ len(test_dataset) }")

    # 5. 保存处理后的训练集和测试集（JSONL格式）
    pd.DataFrame(train_dataset).to_json(
        config.PROCESSED_DIR / "indexed_train.jsonl", orient="records", lines=True
    )
    pd.DataFrame(test_dataset).to_json(
        config.PROCESSED_DIR / "indexed_test.jsonl", orient="records", lines=True
    )

    print("数据处理完成")


if __name__ == "__main__":
    process()
