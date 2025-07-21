#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本

用于快速验证RNN项目的基本功能是否正常工作
运行时间约2-3分钟，适合快速测试

使用方法:
    python quick_test.py
"""

import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')

from data_processor import DataProcessor
from rnn_model import SimpleRNN, RNNTrainer

def quick_test():
    """
    快速测试函数
    """
    print("🚀 RNN项目快速测试")
    print("=" * 40)
    
    try:
        # 1. 测试数据处理
        print("\n1️⃣ 测试数据处理模块...")
        processor = DataProcessor(sequence_length=10)
        data = processor.generate_sample_data(n_samples=500)
        scaled_data = processor.normalize_data()
        X, y = processor.create_sequences()
        X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.2)
        
        print(f"   ✅ 数据生成成功: {len(data)} 个数据点")
        print(f"   ✅ 序列创建成功: {X.shape[0]} 个序列")
        print(f"   ✅ 数据分割成功: 训练集 {len(X_train)}, 测试集 {len(X_test)}")
        
        # 2. 测试模型创建
        print("\n2️⃣ 测试模型创建...")
        model = SimpleRNN(input_size=1, hidden_size=32, num_layers=1)
        trainer = RNNTrainer(model)
        
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   ✅ 模型创建成功: {total_params} 个参数")
        print(f"   ✅ 训练器创建成功: 使用设备 {trainer.device}")
        
        # 3. 测试快速训练
        print("\n3️⃣ 测试模型训练（快速模式）...")
        trainer.train(
            X_train, y_train, X_test, y_test,
            epochs=20,  # 减少训练轮数
            batch_size=16,
            learning_rate=0.01,
            patience=5
        )
        print("   ✅ 模型训练完成")
        
        # 4. 测试预测
        print("\n4️⃣ 测试模型预测...")
        predictions = trainer.predict(X_test)
        print(f"   ✅ 预测完成: 生成 {len(predictions)} 个预测值")
        
        # 5. 测试评估
        print("\n5️⃣ 测试模型评估...")
        metrics, _ = trainer.evaluate(X_test, y_test)
        print("   ✅ 模型评估完成")
        print(f"   📊 RMSE: {metrics['RMSE']:.6f}")
        print(f"   📊 MAE: {metrics['MAE']:.6f}")
        
        # 6. 测试数据反归一化
        print("\n6️⃣ 测试数据反归一化...")
        original_pred = processor.inverse_transform(predictions)
        original_test = processor.inverse_transform(y_test)
        print(f"   ✅ 反归一化完成")
        print(f"   📊 预测值范围: [{original_pred.min():.3f}, {original_pred.max():.3f}]")
        print(f"   📊 真实值范围: [{original_test.min():.3f}, {original_test.max():.3f}]")
        
        # 测试成功
        print("\n" + "=" * 40)
        print("🎉 所有测试通过！项目运行正常")
        print("💡 现在可以运行 'python rnn_demo.py' 查看完整演示")
        print("=" * 40)
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        print("\n🔧 请检查:")
        print("   1. 是否安装了所有依赖包 (pip install -r requirements.txt)")
        print("   2. Python版本是否兼容 (推荐3.7+)")
        print("   3. PyTorch是否正确安装")
        return False

def check_dependencies():
    """
    检查依赖包是否正确安装
    """
    print("🔍 检查依赖包...")
    
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pandas': 'Pandas',
        'sklearn': 'Scikit-learn',
        'matplotlib': 'Matplotlib'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            else:
                __import__(package)
            print(f"   ✅ {name} 已安装")
        except ImportError:
            print(f"   ❌ {name} 未安装")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请运行: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ 所有依赖包已正确安装")
        return True

def system_info():
    """
    显示系统信息
    """
    print("💻 系统信息:")
    print(f"   Python版本: {torch.__version__ if hasattr(torch, '__version__') else 'Unknown'}")
    print(f"   PyTorch版本: {torch.__version__}")
    print(f"   CUDA可用: {'是' if torch.cuda.is_available() else '否'}")
    if torch.cuda.is_available():
        print(f"   CUDA设备数量: {torch.cuda.device_count()}")
        print(f"   当前CUDA设备: {torch.cuda.get_device_name(0)}")
    print(f"   NumPy版本: {np.__version__}")

if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 显示系统信息
    system_info()
    print()
    
    # 检查依赖
    if check_dependencies():
        print()
        # 运行快速测试
        success = quick_test()
        
        if success:
            print("\n🚀 下一步:")
            print("   • 运行完整演示: python rnn_demo.py")
            print("   • 查看项目文档: README.md")
            print("   • 自定义参数: 编辑 rnn_demo.py 中的配置")
    else:
        print("\n❌ 请先安装缺少的依赖包")