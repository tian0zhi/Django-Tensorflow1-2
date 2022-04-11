# import tensorflow as tf
import tensorflow.compat.v1 as tf
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight(shape, regularizer):# 按照形状和正则化系数生成权重Wi和对应正则化
	w = tf.Variable(tf.truncated_normal(shape,stddev=0.1))
	if regularizer != None: tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
	return w

def get_bias(shape):# 按照形状生成阈值Bi
	b = tf.Variable(tf.zeros(shape))
	return b
	
def forward(x, regularizer):# 构建BP网络
	w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
	b1 = get_bias([LAYER1_NODE])
	y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

	w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
	b2 = get_bias([OUTPUT_NODE])
	y = tf.matmul(y1, w2) + b2
	return y
