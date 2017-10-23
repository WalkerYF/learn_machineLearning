### current bpython session - make changes and save to reevaluate session.
### lines beginning with ### will be ignored.
### To return to bpython without reevaluating make no changes to this file
### or save an empty file.
import tensorflow as tf
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)
### Traceback (most recent call last):
###   File "<input>", line 1, in <module>
###     output = tf.mul(input1, input2)
### AttributeError: module 'tensorflow' has no attribute 'mul'
output = tf.multiply(input1, input2)
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],iput2:[2.0]}))
    

### Traceback (most recent call last):
###   File "<input>", line 2, in <module>
###     print(sess.run(output, feed_dict={input1:[7.],iput2:[2.0]}))
### NameError: name 'iput2' is not defined
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.0]}))
###   File "<bpython-input-11>", line 1
###     print(sess.run(output, feed_dict={input1:[7.],input2:[2.0]}))
###     ^
### IndentationError: unexpected indent
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1:[7.],input2:[2.0]}))
    

### [ 14.]
### 
