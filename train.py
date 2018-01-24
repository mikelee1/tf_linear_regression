import tensorflow as tf
import numpy as np

x = np.linspace(-1,1,11)
y_rel = 2*x+0.3
x_data = tf.placeholder('float')
y_data = tf.placeholder('float')
w = tf.Variable(0.0)
b = tf.Variable(0.0)

y = tf.add(tf.mul(w,x_data),b)

loss = tf.square(y_data-y)
op = tf.train.GradientDescentOptimizer(0.01)
train_op = op.minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver()
    for i in range(100):
        sess.run(train_op,feed_dict = {x_data:x,y_data:y_rel})
    
    saver.save(sess,'tmp.ckpt')
    pred = sess.run(y,feed_dict = {x_data:[1]})
    print(pred)
    print(sess.run(w))
    print(sess.run(b))
