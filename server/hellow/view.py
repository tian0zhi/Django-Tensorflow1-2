from django.http import HttpResponse
from django.shortcuts import render
import numpy as np
from PIL import Image
import tensorflow.compat.v1 as tf
from . import mnist_backward
from . import mnist_forward



def restore_model(testPicArr):
	with tf.Graph().as_default() as tg:
		x = tf.placeholder(tf.float32, [None, mnist_forward.INPUT_NODE])
		y = mnist_forward.forward(x, None)
		preValue = tf.argmax(y, 1)

		variable_averages = tf.train.ExponentialMovingAverage(mnist_backward.MOVING_AVERAGE_DECAY)
		variables_to_restore = variable_averages.variables_to_restore()
		saver = tf.train.Saver(variables_to_restore)

		with tf.Session() as sess:
			ckpt = tf.train.get_checkpoint_state(mnist_backward.MODEL_SAVE_PATH)
			if ckpt and ckpt.model_checkpoint_path:
				saver.restore(sess, ckpt.model_checkpoint_path)

				preValue = sess.run(preValue, feed_dict={x:testPicArr})
				return preValue
			else:
				print("No checkpoint file found")
				return -1


def hello(request):
    if request.POST:
        # keysList = list(request.POST.keys())
        getData = list(request.POST.values())[0]
        splitData = getData.split(',')
        # response = HttpResponse('POST:Hello world!!'+str(list(request.POST.values())[0]))
        np_data = np.array(splitData, dtype = np.float32)
        b = np.resize(np_data,(280,280))
        b_int = b.astype(np.uint8)
        im = Image.fromarray(b_int)
        im = im.resize((28,28),Image.ANTIALIAS)
        resdata = np.array(im,dtype = np.float32)/255.0
        resdata = np.resize(resdata,(1,28*28))
        response = HttpResponse(str(restore_model(resdata)))
    elif request.GET:
        # getData = list(request.POST.values())[0]
        # splitData = getData.split(',')
        # response = HttpResponse('POST:Hello world!!'+str(list(request.POST.values())[0]))
        # np_data = np.array(splitData, dtype = np.float32)
        # b = np.resize(np_data,(280,280))
        # b_int = b.astype(np.uint8)
        # im = Image.fromarray(b_int)
        # im = im.resize((28,28),Image.ANTIALIAS)
        response = HttpResponse('GET:success!!')
    else:
        response = HttpResponse('Other:Hello world!!')
    response["Access-Control-Allow-Origin"] = "*"
    response["Access-Control-Allow-Methods"] = "POST, GET,OPTIONS"
    response["Access-Control-Max-Age"] = "1000"
    response["Access-Control-Allow-Headers"] = "*"
    #print(resquest)
    return response

def runoob(request):
    context = {}
    context['hello'] = 'hello world!!'
    return render(request,'runoob.html',context)
