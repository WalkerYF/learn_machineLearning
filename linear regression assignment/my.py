# 多元线性回归
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def linear_model(arg, x_data):
    ''' 输入 线性模型的参数（385*1） 还有x矩阵（20000 * 385） '''
    import numpy as np
    y_data = np.dot(x_data, arg)
    return y_data


def loss(model, arg, x_data, y_data):
    ''' 计算误差， 均方误差，将x_data输入到model得到预测值与y_data进行对比 '''
    ys = model(arg, x_data)
    l = np.sum( np.square(ys - y_data)) / (2 * ys.shape[0])
    return l

def grad(ys, y_data, x_data, beta, beta_size, learn_rate):# 计算梯度
    # 由于计算梯度的过程中，恰好有一部分与y的预测值，因此直接将ys传进来，降低计算量
    g = np.zeros([beta_size, 1])
    for i in range(0, beta_size):
        xi = x_data[:, i].T
        xi = xi[:,np.newaxis]
        # print (xi)
        g[i] = np.sum( (ys-y_data ) * xi ) / ys.shape[0]
    g = g * learn_rate
    return g


# 读取数据
ori_data = pd.read_csv('train.csv')
data = ori_data.as_matrix()
temp_r1 = data[0:1, 1:385]
x = data[0:20000, 1:385]
x_data = np.hstack((np.ones([20000,1]), x)) # 在x数据前加一列1
y_data = data[0:20000, 385:386]

# y = b + b1x1 + b2x2 + ... + bnxn

# 初始化参数
# beta = np.ones([385,1])
beta = np.loadtxt("beta.txt")
beta = beta[:,np.newaxis]

# 训练
for i in range(2000):
    ys = linear_model(beta, x_data)
    g = grad(ys, y_data, x_data, beta, beta.shape[0], 0.02)
    beta = beta - g
    print(loss(linear_model, beta, x_data, y_data))


np.savetxt("beta.txt", beta)