import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function = None):
# 添加一层神经元,输入信号规模为in_size, 输出信号规模微out_size,输入信号为inputs
	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size], name="W"))
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.matmul(inputs, Weights) + biases
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		return outputs 


# 设置数据,创造数据
x_data = np.linspace(-1,1,300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data) - 0.5 + noise

# 占位符
with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
	ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# 输入层-隐藏层
l1 = add_layer(xs, 1, 10, activation_function = tf.nn.relu)
# 隐藏层-输出层
prediction = add_layer(l1, 10, 1, activation_function = None)
# 误差 最小二乘法
with tf.name_scope('loss'):
	loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), reduction_indices=[1]))
# 如何训练:使用梯度下降法,最小化误差
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# 数据可视化
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()

# plt.show() # 会暂停

# 初始化变量
init = tf.global_variables_initializer()
with tf.Session() as sess:
	writer = tf.summary.FileWriter("logs/", sess.graph)# 结构可视化,输出到文件
	# writer = tf.train.SummaryWriter("logs/", sess.graph)# 结构可视化,输出到文件
	sess.run(init)
	for i in range(5000):
		sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
		if i % 50 == 0:
			# print(sess.run(loss,feed_dict={xs: x_data, ys: y_data}))
			# 忽略第一次的错误
			try:
				ax.lines.remove(lines[0])
			except Exception:
				pass
			prediction_value = sess.run(prediction, feed_dict={xs: x_data})
			lines = ax.plot(x_data, prediction_value, 'r-', lw = 5)
			plt.pause(0.1)
			# 数据可视化