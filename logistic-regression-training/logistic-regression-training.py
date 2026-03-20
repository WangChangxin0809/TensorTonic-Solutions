import numpy as np


def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def predict(X, w, b):
    """
    逻辑回归预测函数（输出概率）
    :param X: 特征矩阵 (n_samples, n_features)
    :param w: 权重 (n_features, 1)
    :param b: 偏置 标量
    :return: 预测概率 (n_samples, 1)
    """
    z = np.dot(X, w) + b
    return _sigmoid(z) 


def loss(X, w, b, y):
    """计算交叉熵损失（带防溢出）"""
    predict_y = predict(X, w, b)
    # 修正：防止log(0)溢出，限制predict_y在1e-8~1-1e-8之间
    predict_y = np.clip(predict_y, 1e-8, 1 - 1e-8)
    n = len(X)
    l = -np.mean(y * np.log(predict_y) + (1 - y) * np.log(1 - predict_y))
    return l


def train_logistic_regression(X, y, lr=0.1, steps=1000):
    X = np.array(X, dtype=np.float64)
    y = np.array(y, dtype=np.float64)  # 一维数组 (n_samples,)
    
    n_samples, n_features = X.shape
    w = np.zeros(n_features)  # 改为一维数组 (n_features,)，简化运算
    b = 0.0  # 标量
    
    for i in range(steps):
        predict_y = predict(X, w, b)  # (n_samples,) 一维数组
        # 梯度计算（结果均为标量/一维数组，无维度问题）
        dw = np.dot(X.T, (predict_y - y)) / n_samples  # (n_features,)
        db = np.mean(predict_y - y)  # 标量
        
        # 参数更新（维度匹配，无异常）
        w -= lr * dw
        b -= lr * db
        
        if i % 100 == 0:
            current_loss = loss(X, w, b, y)
            print(f"Step {i}, Loss: {current_loss:.4f}")

    return w, b