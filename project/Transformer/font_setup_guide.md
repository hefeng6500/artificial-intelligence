# 中文字体显示问题解决指南

## 问题描述
在可视化图表中，中文文字显示为方框（□□□）或乱码，这是因为 matplotlib 默认字体不支持中文字符。

## 解决方案

### 方案一：使用系统自带字体（推荐）

#### Windows 系统
```python
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
```

#### macOS 系统
```python
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC']
matplotlib.rcParams['axes.unicode_minus'] = False
```

#### Linux 系统
```python
import matplotlib.pyplot as plt
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False
```

### 方案二：下载并安装字体

#### 1. 下载字体文件
- **SimHei（黑体）**：适用于 Windows
- **Microsoft YaHei（微软雅黑）**：适用于 Windows
- **Noto Sans CJK**：跨平台中文字体

#### 2. 安装字体

**Windows：**
1. 下载字体文件（.ttf 格式）
2. 右键点击字体文件，选择"安装"
3. 或者将字体文件复制到 `C:\Windows\Fonts\` 目录

**macOS：**
1. 下载字体文件
2. 双击字体文件，点击"安装字体"
3. 或者将字体文件复制到 `~/Library/Fonts/` 目录

**Linux：**
```bash
# Ubuntu/Debian
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei

# 或者手动安装
mkdir -p ~/.fonts
cp your-font.ttf ~/.fonts/
fc-cache -fv
```

#### 3. 重启 Python 环境
安装字体后，需要重启 Python 内核或重新启动 Jupyter Notebook。

### 方案三：检查可用字体

```python
import matplotlib.font_manager as fm

# 查看所有可用字体
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(char in f for char in ['黑体', 'SimHei', 'YaHei', 'Microsoft'])]
print("可用的中文字体：", chinese_fonts)

# 查看字体文件路径
for font in fm.fontManager.ttflist:
    if 'SimHei' in font.name or 'YaHei' in font.name:
        print(f"字体名称: {font.name}, 路径: {font.fname}")
```

### 方案四：临时解决方案

如果无法安装字体，可以使用英文标签：

```python
# 使用英文标签
src_tokens = ['I', 'love', 'learning', 'Transformer', 'model']
tgt_tokens = ['<start>', 'I', 'love', 'learning', 'Transformer']

# 或者使用拼音
src_tokens = ['wo', 'ai', 'xuexi', 'Transformer', 'moxing']
```

## 验证字体设置

运行以下代码验证字体设置是否成功：

```python
import matplotlib.pyplot as plt
import matplotlib

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 测试中文显示
plt.figure(figsize=(8, 6))
plt.plot([1, 2, 3, 4], [1, 4, 2, 3])
plt.title('中文标题测试')
plt.xlabel('横轴标签')
plt.ylabel('纵轴标签')
plt.show()

# 如果中文正常显示，说明设置成功
print("字体设置成功！")
```

## 常见问题

### Q1: 设置字体后仍然显示方框
**A1:** 尝试以下步骤：
1. 重启 Python 内核
2. 清除 matplotlib 缓存：`rm -rf ~/.matplotlib/`
3. 确认字体已正确安装

### Q2: 在 Jupyter Notebook 中设置无效
**A2:** 在每个 notebook 的开头添加字体设置代码，或者重启 Jupyter 服务器。

### Q3: Linux 系统中文显示问题
**A3:** 安装中文字体包：
```bash
sudo apt-get install language-pack-zh-hans
sudo apt-get install fonts-wqy-microhei
```

## 本项目中的解决方案

本项目已经在以下文件中添加了字体设置：

1. **transformer_demo.ipynb** - Jupyter notebook 中的字体设置
2. **attention_visualization.py** - 可视化模块中的字体设置

这些设置会自动尝试使用系统中可用的中文字体，如果都不可用会显示警告信息。

## 推荐字体

- **SimHei（黑体）**：Windows 系统推荐
- **Microsoft YaHei（微软雅黑）**：Windows 系统推荐
- **Arial Unicode MS**：macOS 系统推荐
- **Noto Sans CJK**：跨平台推荐
- **WenQuanYi Micro Hei**：Linux 系统推荐

按照以上指南操作，应该能够解决中文字体显示问题。如果仍有问题，请检查系统字体安装情况或使用英文标签作为临时解决方案。