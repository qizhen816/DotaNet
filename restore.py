import tensorflow as tf
from dataset.Reader import Reader
from config import *
from network.network import Network


def feed_data( data, label, idx, keep_prob, learning_rate):
    batch_start = idx * FLAGS.ebatch
    batch_end = min(FLAGS.ebatch + batch_start, FLAGS.train_length)
    if keep_prob != 1.0:
        is_training = True,
    else:
        is_training = False,
    return {
        x: data[batch_start:batch_end],
        y: label[batch_start:batch_end],
        keep_prob: keep_prob,
        # self.net.is_training: is_training,
        # self.learning_rate: learning_rate
    }


x = tf.placeholder(tf.float32, [None, 784])
y_=tf.placeholder(tf.int32,[None,])

dense1 = tf.layers.dense(inputs=x,
                      units=1024,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
dense2= tf.layers.dense(inputs=dense1,
                      units=512,
                      activation=tf.nn.relu,
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                      kernel_regularizer=tf.nn.l2_loss)
logits= tf.layers.dense(inputs=dense2,
                        units=10,
                        activation=None,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        kernel_regularizer=tf.nn.l2_loss)

loss=tf.losses.sparse_softmax_cross_entropy(labels=y_,logits=logits)
train_op=tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_prediction = tf.equal(tf.cast(tf.argmax(logits,1),tf.int32), y_)
acc= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

is_train=True
saver=tf.train.Saver(max_to_keep=3)

#训练阶段
if is_train:
    max_acc=0
    f=open('ckpt/acc.txt','w')
    for i in range(100):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_op, feed_dict={x: batch_xs, y_: batch_ys})
      val_loss,val_acc=sess.run([loss,acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
      print('epoch:%d, val_loss:%f, val_acc:%f'%(i,val_loss,val_acc))
      f.write(str(i+1)+', val_acc: '+str(val_acc)+'\n')
      if val_acc>max_acc:
          max_acc=val_acc
          saver.save(sess,'ckpt/mnist.ckpt',global_step=i+1)
    f.close()

#验证阶段
else:
    model_file=tf.train.latest_checkpoint('ckpt/')
    saver.restore(sess,model_file)
    val_loss,val_acc=sess.run([loss,acc], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print('val_loss:%f, val_acc:%f'%(val_loss,val_acc))
sess.close()