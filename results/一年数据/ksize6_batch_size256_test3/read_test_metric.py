import numpy as np

metrics_info = [
    ('MAE', False),
    ('MSE', False),
    ('RMSE', False),
    ('MAPE', False),
    ('MSPE', False),
    ('RSE', False),
    ('CORR', True),
    ('Spearman', True),
    ('Euclidean', False),
    ('DTW', False),
    ('R2', True),
    ('Accuracy', True)
]


def get_best_value(values, is_larger_better):
    """Get the optimal value"""
    if isinstance(values, (int, float, np.number)):
        return values, 0

    if is_larger_better:
        best_idx = np.argmax(values)
        return values[best_idx], best_idx
    else:
        best_idx = np.argmin(values)
        return values[best_idx], best_idx


def print_metrics():
    try:
        metrics = np.load('test_metrics.npy', allow_pickle=True)

        for i, (name, is_larger_better) in enumerate(metrics_info):
            if i >= len(metrics):
                continue

            metric_values = metrics[i]
            best_value, _ = get_best_value(metric_values, is_larger_better)
            print(f"{name}: {best_value:.6f}")

    except Exception as e:
        print(f"Error reading file: {str(e)}")


if __name__ == "__main__":
    print_metrics()