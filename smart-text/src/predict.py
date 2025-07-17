import torch

import config
from model import InputMethodModel
from tokenizer import JiebaTokenizer


def predict_batch(model, input_tensor):
    """
    批量预测
    :param model: 模型
    :param input_tensor: 输入张量 [batch_size, sql_len]
    :return: [[1,2,3,4,5],[6,7,8,9,10]]
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        # outputs.shape: [batch_size, vocab_size]
        top5_indexes = torch.topk(outputs, k=5).indices
        # top5_indexes.shape: [batch_size, 5]

    top5_indexes_list = top5_indexes.tolist()
    return top5_indexes_list


def predict(text, model, tokenizer, device):
    # 数据
    index_list = tokenizer.encode(text)
    input_tensor = torch.tensor([index_list]).to(device)
    # input_tensor.shape: [batch_size, seq_len]

    top5_indexes_list = predict_batch(model, input_tensor)

    top5_words = [tokenizer.index2word[index] for index in top5_indexes_list[0]]
    return top5_words


def run_predict():
    # 加载资源
    ##############################################
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'设备:{device}')

    # 创建tokenizer
    tokenizer = JiebaTokenizer.from_vocab(config.PROCESSED_DIR / 'vocab.txt')

    # 模型
    model = InputMethodModel(vocab_size=tokenizer.vocab_size).to(device)
    model.load_state_dict(torch.load(config.MODELS_DIR / 'model.pt'))
    ##############################################

    print("请输入下一个词:(输入q或者quit退出)")
    history_input = ''
    while True:
        user_input = input("> ")
        if user_input in ['q', 'quit']:
            print("程序已退出")
            break
        if user_input.strip() == '':
            print("请输入下一个词")
            continue
        history_input += user_input
        print(f"历史输入：{history_input}")
        top5_words = predict(history_input, model, tokenizer, device)
        print(f"预测结果：{top5_words}")


if __name__ == '__main__':
    run_predict()
