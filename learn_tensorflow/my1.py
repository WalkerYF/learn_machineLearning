### current bpython session - make changes and save to reevaluate session.
### lines beginning with ### will be ignored.
### To return to bpython without reevaluating make no changes to this file
### or save an empty file.
import numpy as np
import tensorflow as tf
# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3
import matplotlib as plib
import matplotlib.pyplot as plt
plt.scatter(x_data,y_data)
### <matplotlib.collections.PathCollection object at 0x7f5d9cabacc0>
plt.show()
# create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
### WARNING:tensorflow:From /usr/local/lib/python3.5/dist-packages/tensorflow/
### python/util/tf_should_use.py:112: initialize_all_variables (from tensorflo
### w.python.ops.variables) is deprecated and will be removed after 2017-03-02
### .
### Instructions for updating:
### 
### Use `tf.global_variables_initializer` instead.
init = tf.global_variables_initializer()
# create end
sess = tf.Session()
sess.run(init)
   # important
for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights),sess.run(biases))
        
    

### 0 [ 0.33897763] [ 0.23312712]
### 20 [ 0.15244269] [ 0.27232221]
### 40 [ 0.11282052] [ 0.29323369]
### 60 [ 0.10313419] [ 0.29834586]
### 80 [ 0.10076621] [ 0.29959562]
### 100 [ 0.10018731] [ 0.29990116]
### 120 [ 0.1000458] [ 0.29997584]
### 140 [ 0.10001121] [ 0.29999411]
### 160 [ 0.10000274] [ 0.29999855]
### 180 [ 0.10000069] [ 0.29999965]
### 200 [ 0.10000016] [ 0.29999992]
### 
