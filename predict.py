import tensorflow as tf

x_data = tf.placeholder('float')
w = tf.Variable(0.0)
b = tf.Variable(0.0)
y = tf.add(tf.mul(w,x_data),b)

saver = tf.train.Saver()
with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state('/home/tensorflow_learn/trials/')
    saver.restore(sess,ckpt.model_checkpoint_path)
    pred = sess.run(y,feed_dict = {x_data:[1]})
    print(pred)
    print(sess.run(w))
    print(sess.run(b))

