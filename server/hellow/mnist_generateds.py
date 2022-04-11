#coding:utf-8
import tensorflow as tf
import numpy as np
from PIL import Image
import os

image_train_path='./mnist_data_jpg/mnist_train_jpg_60000/'
label_train_path='./mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train='./data/mnist_train.tfrecords'
image_test_path='./mnist_data_jpg/mnist_test_jpg_10000/'
label_test_path='./mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test='./data/mnist_test.tfrecords'
data_path='./data'
resize_height = 28
resize_width = 28

def write_tfRecord(tfRecordName, image_path, label_path):
	# 写record tfRecordName:记录名 image_path:图片路径 label_path:标签路径+文件名
	writer = tf.python_io.TFRecordWriter(tfRecordName)
	# 可以看做初始化一个文件输入流(java)
	num_pic = 0 
	f = open(label_path, 'r')# 在一个txt文件中有图片名和对应的数值
	contents = f.readlines()#读取整个文件内容
	f.close()
	for content in contents:
		value = content.split()
		img_path = image_path + value[0] 
		img = Image.open(img_path)
		img_raw = img.tobytes()# 将图片转换成二进制
		labels = [0] * 10# 生成一个 长度为10的全零列表
		labels[int(value[1])] = 1# 对应位置置1，表示是哪个数字
			
		example = tf.train.Example(features=tf.train.Features(feature={
				'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw])),
				'label': tf.train.Feature(int64_list=tf.train.Int64List(value=labels))
				}))
		# 生成实例example，特征feature分别写入字段，字段的值，字段的数据类型——像数据库建表并插入数据
		writer.write(example.SerializeToString())# 序列化写入
		num_pic += 1 
		print ("the number of picture:", num_pic)
	writer.close()
	print("write tfrecord successful")

def generate_tfRecord():
	isExists = os.path.exists(data_path) 
	if not isExists: 
		os.makedirs(data_path)
		print('The directory was created successfully')
	else:
		print('directory already exists' )
	write_tfRecord(tfRecord_train, image_train_path, label_train_path)
	write_tfRecord(tfRecord_test, image_test_path, label_test_path)
	
def read_tfRecord(tfRecord_path):
	# 读取record文件
	filename_queue = tf.train.string_input_producer([tfRecord_path], shuffle=True)
	# shuffle：布尔值。如果为true，则在每个epoch内随机打乱顺序。所以这里置True
	reader = tf.TFRecordReader()
	# 可以看做初始化一个文件输出流(java)
	_, serialized_example = reader.read(filename_queue) 
	features = tf.parse_single_example(serialized_example,
										features={
										'label': tf.FixedLenFeature([10], tf.int64),
										'img_raw': tf.FixedLenFeature([], tf.string)
										})
	# FixedLenFeature 特征数
	img = tf.decode_raw(features['img_raw'], tf.uint8)
	img.set_shape([784])
	img = tf.cast(img, tf.float32) * (1. / 255)
	label = tf.cast(features['label'], tf.float32)
	return img, label 
	  
def get_tfrecord(num, isTrain=True):
	if isTrain:
		tfRecord_path = tfRecord_train
	else:
		tfRecord_path = tfRecord_test
	img, label = read_tfRecord(tfRecord_path)
	img_batch, label_batch = tf.train.shuffle_batch([img, label],
													batch_size = num,
													num_threads = 2,
													capacity = 1000,
													min_after_dequeue = 700)
	# 会从总样本([img, label])中顺序取出(capacity)1000组数据，打乱顺序，每次输出(batch_size)num组.
	# 如果capacity小于min_after_dequeue,会从总样本中提取数据直至min_adter_dequeue.
	# 整个过程用到了(num_threads)2个线程
	return img_batch, label_batch

def main():
	generate_tfRecord()

if __name__ == '__main__':
	main()
