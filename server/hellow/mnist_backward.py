# import tensorflow as tf
import tensorflow.compat.v1 as tf
# from tensorflow.examples.tutorials.mnist import input_data
from . import mnist_forward#前向传播
import os
from . import mnist_generateds#1 数据集生成和读取

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="/home/ubuntu/python/website/hellow/hellow/model/"
MODEL_NAME="mnist_model"
train_num_examples = 60000#2 总训练样本数

def backward():

	x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
	y_ = tf.placeholder(tf.float32, [None, mnist_forward.OUTPUT_NODE])
	y = mnist_forward.forward(x, REGULARIZER)# 前向传播
	global_step = tf.Variable(0, trainable=False) 

	ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
	# 交叉熵的损失函数
	cem = tf.reduce_mean(ce)
	loss = cem + tf.add_n(tf.get_collection('losses'))
	# 加正则化的损失函数

	learning_rate = tf.train.exponential_decay(
		LEARNING_RATE_BASE,#初始学习率
		global_step,# 训练轮数
		train_num_examples / BATCH_SIZE, # Learn_rate_step是多少轮更新一次学习率，他一般设为总样本数/batch_size 
		LEARNING_RATE_DECAY,# 下降率
		staircase=True)
	# 指数衰减学习率
	train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

	ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
	ema_op = ema.apply(tf.trainable_variables())
	# 滑动平均
	with tf.control_dependencies([train_step, ema_op]):
		train_op = tf.no_op(name='train')
	# 将滑动平均和训练节点结合成一个训练节点
	write = tf.summary.FileWriter("log",tf.get_default_graph())
	write.close()
	saver = tf.train.Saver()#初始化模型保存
	
	img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)#3
	# 获取一定规则的数据(类)

	with tf.Session() as sess:
		init_op = tf.global_variables_initializer()# 初始化全局变量
		sess.run(init_op)# 执行

		ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
		# 如果指定路径有模型，则引入
		
		coord = tf.train.Coordinator()#4
		# 线程协调器，它会将 样本与标签批获取的操作放在这里，提高效率
		threads = tf.train.start_queue_runners(sess=sess, coord=coord)#5
		
		for i in range(STEPS):
			xs, ys = sess.run([img_batch, label_batch])#6 按照tf.train.shuffle_batch的规则返回数据
			# 从这里可以看出mnist_generateds.get_tfrecord 返回的img_batch与label_batch并不是类似于dict的数据集合
			# 而是一种包含数据的数据封装，像java中的int与Integer
			_, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
			# 喂入数据
			if i % 100 == 0:
				print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
				saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
				# 保存模型

		coord.request_stop()#7
		coord.join(threads)#8
		# 线程协调器 结束


def main():
	backward()#9

if __name__ == '__main__':
	main()


