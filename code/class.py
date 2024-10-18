import os
import pandas as pd
import shutil

# 定义路径
labels_file = 'D:/imagenet_train_10000/dev.csv'  # 标签文件的路径
image_dir = 'D:/imagenet_train_10000/val'  # 图像原始路径
target_dir = 'D:/imagenet_train_10000/val'  # 目标路径（同样是val文件夹）

# 读取标签文件
labels = pd.read_csv(labels_file)

# 创建文件夹并分类图像
for label in labels['TrueLabel'].unique():
    # 创建每个类别的文件夹
    os.makedirs(os.path.join(target_dir, str(label)), exist_ok=True)

# 遍历标签，将图像移动到对应的文件夹
for index, row in labels.iterrows():
    image_id = row['ImageId']
    true_label = row['TrueLabel']

    # 源文件路径
    source_path = os.path.join(image_dir, image_id)
    # 目标文件夹路径
    target_path = os.path.join(target_dir, str(true_label), image_id)

    # 复制文件到目标文件夹
    shutil.copy(source_path, target_path)

print("图像分类完成！")
