import pandas as pd
import os


def split_dataset(data_path):
    # 读取原始数据集
    df = pd.read_csv(data_path)

    # 计算划分点
    num_total = len(df)
    num_train = int(num_total * 0.8)
    num_test = int(num_total * 0.2)

    # 划分数据集
    train_df = df[:num_train]
    test_df = df[-num_test:]

    # 构造输出路径（与原数据集在同一目录）
    directory = os.path.dirname(data_path)
    train_path = os.path.join(directory, 'train.csv')
    test_path = os.path.join(directory, 'test.csv')

    # 保存数据集
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    # 打印信息
    print(f"\n数据集已成功划分：")
    print(f"原始数据集大小: {len(df)} 条记录")
    print(f"训练集大小: {len(train_df)} 条记录")
    print(f"测试集大小: {len(test_df)} 条记录")
    print(f"\n文件已保存：")
    print(f"训练集: {train_path}")
    print(f"测试集: {test_path}")


if __name__ == "__main__":
    # 设置数据集路径
    data_path = "data/ETT/QianTangRiver2020-2024WorkedFull.csv"

    # 执行划分
    split_dataset(data_path)