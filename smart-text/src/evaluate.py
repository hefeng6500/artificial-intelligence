import torch
import config
from model import InputMethodModel
from dataset import get_dataloader
from predict import predict_batch
from tokenizer import JiebaTokenizer


def evaluate_model(model, dataloader, device):
    total_count = 0
    top1_acc_count = 0
    top5_acc_count = 0
    for inputs, targets in dataloader:
        inputs = inputs.to(device)

        targets = targets.tolist()
        # targets = [5,8,13]
        top5_indexes_list = predict_batch(model, inputs)
        # top5_indexes_list = [[5,8,13,14,15],[5,8,13,14,15],[5,8,13,14,15]]

        for target, top5_indexes in zip(targets, top5_indexes_list):
            # target=5
            # top5_indexes=[5,8,13,14,15]
            total_count += 1
            if target == top5_indexes[0]:
                top1_acc_count += 1
            if target in top5_indexes:
                top5_acc_count += 1
    return top1_acc_count / total_count, top5_acc_count / total_count


def run_evaluate():
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

    # 数据集
    dataloader = get_dataloader(train=False)
    ##############################################

    # 评估模型
    top1_acc, top5_acc = evaluate_model(model, dataloader, device)
    print(f'========== 评估结果 ==========')
    print(f'top1准确率:{top1_acc:.4f}')
    print(f'top5准确率:{top5_acc:.4f}')
    print('=============================')


if __name__ == '__main__':
    run_evaluate()
