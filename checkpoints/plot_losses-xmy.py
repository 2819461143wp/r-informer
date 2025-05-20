import json
import matplotlib.pyplot as plt
import numpy as np

# 读取 JSON 文件
def load_losses(filename="losses.json"):
    """从 JSON 文件中加载损失值"""
    with open(filename, 'r') as f:
        losses_dict = json.load(f)
    return losses_dict

# 绘制损失曲线
def plot_losses_from_file(filename="losses.json"):
    """从 JSON 文件中加载损失值并绘制曲线"""
    # 加载损失值
    losses_dict = load_losses(filename)
    train_losses = losses_dict.get("train_losses", [])
    vali_losses = losses_dict.get("vali_losses", [])
    test_losses = losses_dict.get("test_losses", [])

    # 检查是否成功加载数据
    if not train_losses or not test_losses or not vali_losses:
        print("未找到训练、验证或测试损失数据！")
        return
    
    # 找到最优损失值和对应的epoch
    best_train_loss = min(train_losses)
    best_train_epoch = train_losses.index(best_train_loss)
    best_vali_loss = min(vali_losses)
    best_vali_epoch = vali_losses.index(best_vali_loss)
    best_test_loss = min(test_losses)
    best_test_epoch = test_losses.index(best_test_loss)
    
    # 打印最优损失值
    print(f"最优训练损失: {best_train_loss:.4f} (Epoch {best_train_epoch+1})")
    print(f"最优验证损失: {best_vali_loss:.4f} (Epoch {best_vali_epoch+1})")
    print(f"最优测试损失: {best_test_loss:.4f} (Epoch {best_test_epoch+1})")

    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(vali_losses, label='Validation Loss')
    plt.plot(test_losses, label='Test Loss')
    
    # 在图上标注最优点
    plt.plot(best_train_epoch, best_train_loss, 'ro', markersize=8, label=f'Best Train Loss: {best_train_loss:.4f}')
    plt.plot(best_vali_epoch, best_vali_loss, 'bo', markersize=8, label=f'Best Validation Loss: {best_vali_loss:.4f}')
    plt.plot(best_test_epoch, best_test_loss, 'go', markersize=8, label=f'Best Test Loss: {best_test_loss:.4f}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training, Validation and Testing Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.show()

# 示例：读取并绘制损失曲线
if __name__ == "__main__":
    json_file_path = "losses.json"  # 直接使用同目录下的文件
    plot_losses_from_file(json_file_path)