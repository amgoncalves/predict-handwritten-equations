from __future__ import division
from __future__ import print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:06:17 2017

@author: paulcabrera

RUN:
$ source ~/tensorflow/bin/activate
$ python 5-4-17.py ./annotate/
"""

"""
Comments:
 
    
    - We're allowed to submit our model to TAs before the deadline so they can run it and let us know the results.
    
    - maybe try to make some predictions on the examples folder
    
    - More training and test data? MNIST?
    http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/: may have to invert colors and delete folders for symbols that
    don't appear. 
        - An issue with using MNIST is it has more digits than necessary. We just need 0, 1, 2, 3, 4, 6. So maybe filter the
        other digits out
        - equations.pdf:
        - Other symbols: =, +, -, division sign, bar division sign, (, ), pi, sqrt, delta, ..., +- sign
        - Having trouble finding data for the above symbols.
        - letters: x, y, a, b, c, m, n, d, p, k, f, s, i, o, t, A
        
    - We may not have enough data for the more complicated neural network to work properly.
    
    - Perhaps reducing the number of layers would lead to better results
    
    - I increased the batch size to train the model properly given the more complex neural network
    
    - Consider using TA's input_wrapper instead of the various functions I use to transform the image.
    
    - predictions.txt results are far more inaccurate currently using the more complex neural network compared to the simple one
"""

import sys
import glob
from PIL import Image, ImageFilter
import skimage.measure as skm
import scipy.misc as scm
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf
from skimage.transform import resize,warp,AffineTransform

trainpath = sys.argv[-1]  # # path for the folder containg the training images i.e. the path for 'annotated'
tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables

class SymPred():
	# prediction is a string; the other args are ints
	def __init__(self,prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |		|
			   |		|
				--------.
						 (x2,y2)
		"""
		self.prediction = prediction 
		self.x1 = x1
		self.y1 = y1
		self.x2 = x2
		self.y2 = y2
	def __str__(self):
		return self.prediction + '\t' + '\t'.join([
												str(self.x1),
												str(self.y1),
												str(self.x2),
												str(self.y2)])

def padim(im):
	""" Pads image to make it into a square.
	
	Parameters
	----------
	im : ndarray
		An image to be padded.
		
	Returns
	-------
	ndarray
		A copy of im with padding.
	"""
	rows = len(im)
	cols = len(im[0])
	zeros = max(rows, cols) - min(rows, cols)
	left, right, top, bottom  = 0, 0, 0, 0
	if rows > cols:
		left = zeros//2
		right = zeros - left
	elif rows < cols:
		top = zeros//2
		bottom = zeros - top
	return np.pad(im, ((top, bottom), (left, right)), 'constant')

def fullpadim(im):
	""" Pads left, right, bottom, and top with zeros and then do additional padding to make image into a square.
	
	Parameters
	----------
	im : ndarray
		An image to be padded.
		
	Returns
	-------
	ndarray
		A copy of im with padding.
	"""
	rows = len(im)
	cols = len(im[0])
	zeros = max(rows, cols) - min(rows, cols)
	left = zeros//2
	right = zeros - left
	left = right
	bottom = zeros//2
	top = zeros - bottom
	bottom = top
	im = np.pad(im, ((top, bottom), (left, right)), 'constant')
	if len(im) != len(im[0]):
		im = padim(im)
	return im

def cropim(im):
	""" Returns image that has been cropped using a bounding box.
	
	Reference: http://chayanvinayak.blogspot.com/2013/03/bounding-box-in-pilpython-image-library.html
	
	Parameters
	----------
	im : ndarray
		An image to be cropped.
	
	Returns
	-------
	ndarray
		A copy of im cropped using bound box obtained from ???
	"""
	im = Image.fromarray(im)
	z = im.split()
	left,upper,right,lower = z[0].getbbox() 
	#im = (im.crop((left,upper,right,lower))).filter(ImageFilter.SHARPEN) # filter doesn't work for some reason 
	im = (im.crop((left,upper,right,lower)))
	return np.array(im.getdata()).reshape((im.size[1], im.size[0])) # confirmed it's im.size[1] and im.size[0] in that order
	
def normalize(im):
	""" Normalize ndarray to values between 0 and 1
	
	Parameters
	----------
	img : ndarray
		Image data to be normalized.
		
	Returns
	-------
	ndarray
		A normalized copy of im.
	"""
	return im / im.max() # MNIST data says 0 means white and 255 means black. MNIST images are normalized between 0 and 1. 
	
def newim(im):
	""" Returns a normalized and padded square 28x28 pixel copy of an equation component.
	
	Parameters
	----------
	im : ndarray
		Image data.
	
	Returns
	-------
	ndarray
		A normalized, padded, square copy of im.
	
	"""
	return normalize(fullpadim(im))

def connectedcomps(im):
	""" Returns a list of connected components as ndarrays that have more than 50 pixels
	
	Parameters
	----------
	im : ndarray
		Image of an equation.
		
	Returns
	-------
	(ndarray, ndarray)	
		A kist of the equation's components and a list of corresponding bounding box coordinates.
	"""
	comps = skm.regionprops(skm.label(im > 0)) # im > 0 leads to all values greater than 0 becoming True i.e. 1 and all equal to 0 False i.e. 0
	# I am not entirely sure if im > 0 is necessary since I omit components with fewer than 50 pixels in the code below
	# Without the if condition and without im > 0, however, we get an unreasonably high number of components, most of which are useless
	bbcoords = []
	newcomps = []
	for i in range(len(comps)):
		if comps[i].area < 50:
			continue
		bbcoords += [comps[i].bbox]
		newcomps += [normalize(morphology.dilation(
							  scm.imresize(
									fullpadim(cropim(np.asarray(comps[i].image, dtype=np.float32))), 
									(32, 32), 'bicubic')))]
	return (newcomps, bbcoords)	 

def getlocalpath(path):
	""" Returns the last value of a filepath.
	
	Parameters
	----------
	path : string
		A complete image file path.	 Ex: 'path/to/a/file.png'
	
	Returns
	-------
	string
		The containing directory of path.
	"""
	return path.split('/')[-1]

def geteqnpath(path):
	""" Given the full path for a symbol, return the path of the corresponding equation.
	
	Parameters
	----------
	path : string
		A complete image component file path. Ex: '$home/annotated/SKMBT_36317040717260_eq2_sqrt_22_98_678_797.png'
		
	Returns
	-------
	string
		Path of the corresponding equation image.  Ex: '$home/annotated/SKMBT_36317040717260_eq2.png'		
	"""
	s = ""
	count = 0 # keeps track of number of underscores encountered
	for c in path:
		if c == '_':
			count += 1
		if count == 3:
			break
		s += c
	if '.png' in s:
		return s
	return s + '.png'
		

def getdict(folder):
	""" Returns a dictionary where the key is the equation image path and the value is a list of paths for the symbols of the equation.
	
	Parameters
	----------
	folder : string
		The full path of the folder containing the annotated images.
	
	Returns
	-------
	dict(string, list(string))
		A dictionary of image paths keys and component path list values.
	"""
	paths = glob.glob(folder+'/*.png')
	eqns = {}
	d = {}
	iseqn = False
	i = -5
	s = ''
	for p in paths:
		c = p[i] # p[-5], which is the character right before the .png
		# use this loop to see if 'eq' occurs before the first instance of '_' when going in reverse order
		while c != '_' and (not iseqn) and abs(i) <= len(p): 
			s += c
			if 'eq' in s[::-1]: # reverse of s since s is being built up in reverse
				iseqn = True
			i -= 1
			if abs(i) <= len(p):
				c = p[i]
		if iseqn: 
			eqns[p] = []
		else: # path is for an image of a symbol, not equation
			eqnpath = geteqnpath(p)
			if eqnpath in eqns: # otherwise: FileNotFoundError
				if eqnpath not in d:
					d[eqnpath] = []
				d[eqnpath] += [p]
		s = ''
		iseqn = False
		i = -5
	return d
	
def getsypaths(folder):
	d = getdict(folder)
	lst = list(d.values())
	sypaths = []
	for e in lst:
		if e:	# not the empty list
			sypaths += e
	return sypaths

def geteqpaths(folder):
	d = getdict(folder)
	return list(d.keys())

 ### CHANGED: Using a (32, 32) image now and I moved the np.resize function elsewhere ###
def transform(im):
	return normalize(morphology.dilation(scm.imresize(fullpadim(im), (32, 32), 'bicubic')))
 
def geteqims(folder):
	return [(scm.imread(impath), impath) for impath in geteqpaths(folder)]
		 
# Get the images of the symbols. These will be used as training data
# list of tuples: (ndarray length 28*28 of image, imagepath)
def getsyims(folder):
	return [(transform(scm.imread(impath)), impath) for impath in getsypaths(folder)]
			
# given the path for a symbol in the format of images in annotated, extract the label
def getlabel(path):
	# once you get to the 4th underscore as you move backwards through the path, build the string until you reach the 5th underscore
	count = 0 # count of underscores
	label = ''
	i = -1
	while count < 5 and abs(i) <= len(path):
		if path[i] == '_':
			count += 1
		elif count == 4: # assuming '_' is not a valid symbol
			label += path[i]
		i -= 1
	return label[::-1] # reverse
	
# Add the corresponding label to each tuple for the argument trainims, which is the result of getsyims(trainpath)
def addlabel(trainims):
	""" Add the corresponding label to each tuple for the argument trainims, which is the result of getsyims(trainpath).
	
	Parameters
	----------
	trainims : *** type ***
		*** Description of trainims ***
	
	Returns
	-------
	*** return type ***
		*** Description of return type ***
	"""
	return [(im, impath, getlabel(impath)) for (im, impath) in trainims]
	
def unpack(syims):
	""" *** Description here ***
	
	Parameters
	----------
	syims : ** type **
		** Description here. **
	
	Returns
	-------
	(array.**type**, array.**type**, array.**type**)
		ims - 
		paths -
		labels -
	"""
	ims, paths, labels = [], [], []
	for e in syims:
		ims += [e[0]]
		paths += [e[1]]
		labels += [e[2]]
	#return (np.asarray(ims), np.asarray(paths), np.asarray(labels)) # currently seems unnecessary based on what I'm doing in my_next_batch
	return (ims, paths, labels)		   

# args: lst - sorted list of unique labels e.g. labellst = list(set(labels)).sorted()
# returns dictionary of onehot lists for each label
def oneHot(lst):
	""" *** Description ***
	
	Parameters
	----------
	lst : list
		Sorted list of unique labels. e.g. labellst = list(set(labels)).sorted()
		
	Returns
	-------
	dict.***type***
		Dictionary of onehot lists for each label.
	"""
	d = {}	
	n = len(lst)
	onehotlst = [0]*n # list of zeros of length len(lst)
	i = 0
	for label in lst:
		onehotlst[i] = 1
		d[label] = onehotlst
		onehotlst = [0]*n
		i += 1
	return d
	
# return an ndarray of one-hot lists for every element. INCOMPLETE
def oneHotTotal(lst):
	""" Return an ndarray of one-hot lists for every element. INCOMPLETE
	
	Parameters
	----------
	lst : list
		List of component labels.
	
	Returns
	-------
	array.list.
		Array of one-hot lists.
	"""
	arr = np.asarray(oneHot(lst[0]))
	for i in range(1, len(lst)):
		arr = np.vstack((arr, oneHot(lst[i])))
	return arr

syims = addlabel(getsyims(trainpath)) # symbol (not equation) images; result is list of 3-element tuples: 

(trainims, trainpaths, labels) = unpack(syims)
labellst = list(set(labels)) 
labellst.sort() # sorted list of unique labels
onehotdict = oneHot(labellst)

	
# affine transformation
def image_deformation(image):
    random_shear_angl = np.random.random() * np.pi/6 - np.pi/12
    random_rot_angl = np.random.random() * np.pi/6 - np.pi/12 - random_shear_angl
    random_x_scale = np.random.random() * .4 + .8
    random_y_scale = np.random.random() * .4 + .8
    random_x_trans = np.random.random() * image.shape[0] / 4 - image.shape[0] / 8
    random_y_trans = np.random.random() * image.shape[1] / 4 - image.shape[1] / 8
    dx = image.shape[0]/2. \
            - random_x_scale * image.shape[0]/2 * np.cos(random_rot_angl)\
            + random_y_scale * image.shape[1]/2 * np.sin(random_rot_angl + random_shear_angl)
    dy = image.shape[1]/2. \
            - random_x_scale * image.shape[0]/2 * np.sin(random_rot_angl)\
            - random_y_scale * image.shape[1]/2 * np.cos(random_rot_angl + random_shear_angl)
    trans_mat = AffineTransform(rotation=random_rot_angl,
                                translation=(dx + random_x_trans,
                                             dy + random_y_trans),
                                             shear = random_shear_angl,
                                             scale = (random_x_scale,random_y_scale))
    return warp(image,trans_mat.inverse,output_shape=image.shape)
  
# uses variables defined outside of this function: trainims, trainpaths, labellst
### CHANGED ###
def my_next_batch(batch_size=10):
	""" *** Description ***
		
	Parameters
	----------
	trainims : ** type **
		*** Description of trainims ***
	
	Returns
	-------
	(array, array, array)
		batch_x - numpy pixel arrays for each symbol
		batch_y - one hot tensors for each symbol
		batch_z - image path for the symbol's associate equation
	"""
	# randomly pick ten elements from trainims
	size = len(trainims)
	indices = [np.random.randint(0, size) for j in range(batch_size)]
	numlabels = len(labellst)
	batch_x = np.zeros((batch_size, 32*32))
	batch_y = np.zeros((batch_size, numlabels)) # rows = batch_size and cols = # of unique symbols
	batch_z = np.empty((batch_size, 1), dtype='<U150') # this is for image paths. row is for each image and column is 1 because it's just one string
	for j in range(batch_size):
		k = indices[j]
		batch_x[j] = np.asarray(np.reshape(image_deformation(trainims[k]), 32*32))
		batch_y[j] = np.asarray(onehotdict[labels[k]])
		batch_z[j] = np.asarray(trainpaths[k])
	return batch_x, batch_y, batch_z

def batch_norm_layer(inputs, decay = 0.9, trainflag=True):
    is_training = trainflag
    epsilon = 1e-3
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                                             batch_mean, batch_var, beta, scale, epsilon),batch_mean,batch_var
    else:
        return tf.nn.batch_normalization(inputs,
                                         pop_mean, pop_var, beta, scale, epsilon),pop_mean,pop_var
                                    
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.01) # stddev changed from 0.1 to 0.01
    var = tf.Variable(initial) # originally 2nd line was return tf.Variable(initial)
    weight_decay = tf.multiply(tf.nn.l2_loss(var), 1e-5, name='weight_loss')
    tf.add_to_collection('losses', weight_decay) # difference is that weight_decay is a new variable that's added to collection
    return var
	  
def bias_variable(shape):
    initial = tf.constant(0., shape=shape) # original tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W, padding='SAME', stride=1): # unchanged, just default args are added
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding=padding)
    
def max_pool_2x2(x, stride=2): # unchanged, just default arg is added
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, stride, stride, 1], padding='SAME')
def avg_pool_global(x,ks):
    return tf.nn.avg_pool(x, ksize=[1, ks, ks, 1],
                          strides=[1, 1, 1, 1], padding='VALID')
    
sess = tf.InteractiveSession()
   
x = tf.placeholder(tf.float32, shape=[None, 1024]) # changed from None, 784

y_ = tf.placeholder(tf.float32, shape=[None, len(labellst)]) # len(label(lst)) is the number of unique labels

box = tf.placeholder(tf.int32, shape=[None, 4])
name = tf.placeholder(tf.string, shape=[None, 1])
n = tf.placeholder(tf.int32, shape=[None, 1])

padding = 'SAME'

"""
#CONVOLUTION LAYER 1
"""
x_image = tf.reshape(x, [-1,32,32,1])

# instesad of [5, 5, 1, 32] we will compute 8 features for a 3x3 patch; # of input channels, 1, remains the same.
W_conv1 = weight_variable([3, 3, 1, 8]) # changed from [5, 5, 1, 32]
# explanation of the _, _; they're used becaused batch_norm_layer returns three vars but we don't need the last two
tmp_1, _, _ = batch_norm_layer(conv2d(x_image, W_conv1, padding=padding)) # difference of conv2d: no addition of bias variable, b_conv1
h_conv1 = tf.nn.relu(tmp_1) # basically the same if we don't consider batch_norm_layer part 

# the first element batch_norm_layer returns is tf.nn.batch_normalization(conv2d(x, W_conv1, padding=padding), other args)
# https://www.tensorflow.org/api_docs/python/tf/nn/batch_normalization. Basically normalizes a tensor of arbitrary dimensions
# using mean and variance

"""
#CONVOLUTION LAYER 2
"""
# instead of [5, 5, 32, 64] we compute 8 features for a 3x3 patch given 8 input channels.
W_conv2 = weight_variable([3, 3, 8, 8])	
tmp_2, _, _ = batch_norm_layer(conv2d(h_conv1, W_conv2, padding=padding))
h_conv2 = tf.nn.relu(tmp_2+x_image)

#POOL... I guess these three blocks of statements constitute pooling. h_skip2 is used to compute h_conv4
W_skip2 = weight_variable([3,3,8,16])
tmp_skip2, _, _ = batch_norm_layer(conv2d(h_conv2, W_skip2, padding=padding, stride=2))
h_skip2 = tf.nn.relu(tmp_skip2)

"""
#CONVOLUTION LAYER 3
"""
W_conv3 = weight_variable([3, 3, 8, 16])	
tmp_3, _, _ = batch_norm_layer(conv2d(h_conv2, W_conv3, padding=padding, stride=2))
h_conv3 = tf.nn.relu(tmp_3)

"""
#CONVOLUTION LAYER 4
"""
W_conv4 = weight_variable([3, 3, 16, 16])
tmp_4, _, _ = batch_norm_layer(conv2d(h_conv3, W_conv4, padding=padding))
h_conv4 = tf.nn.relu(tmp_4+h_skip2)

#POOL... these three blocks of statements
W_skip4 = weight_variable([1,1,16,32])
tmp_skip4, _, _ = batch_norm_layer(conv2d(h_conv4, W_skip4, padding=padding, stride=2))
h_skip4 = tf.nn.relu(tmp_skip4)

"""
#CONVOLUTION LAYER 5
"""
W_conv5 = weight_variable([3, 3, 16, 32])	
tmp_5, _, _ = batch_norm_layer(conv2d(h_conv4, W_conv5, padding=padding, stride=2))
h_conv5 = tf.nn.relu(tmp_5)

"""
#CONVOLUTION LAYER 6
"""
W_conv6 = weight_variable([3, 3, 32, 32])
tmp_6, _, _ = batch_norm_layer(conv2d(h_conv5, W_conv6,padding=padding))
h_conv6 = tf.nn.relu(tmp_6+h_skip4)

"""
#POOLING
"""
h_pool6 = avg_pool_global(h_conv6,8)

"""
#DENSELY CONVOLUTED LAYER
"""	   
W_fc1 = weight_variable([1,1,32,64])
b_fc1 = bias_variable([64])
h_fc1 = conv2d(h_pool6, W_fc1,padding=padding) + b_fc1
	
"""
#DROPOUT
"""
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
"""
#READOUT LAYER
"""
W_readout = weight_variable([1,1,64, len(labellst)])
b_readout = bias_variable([len(labellst)])		
readout = conv2d(h_fc1, W_readout, padding=padding) + b_readout
h_fc1 = h_conv6
	
"""
#TRAIN & EVALUATE
"""
trainflag = True
if trainflag:
    readout = tf.reshape(readout, [-1,len(labellst)])
    #y_conv=tf.nn.softmax(readout)
    y_conv = readout
    y_res = tf.argmax(y_conv,1)
    W_conv1 = W_conv1
    trainflag = False
else:
    readout = readout

identitybox = tf.identity(box)
identityname = tf.identity(name)
identitynum = tf.identity(n)

cross_entropy_mean = tf.reduce_mean(
	tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
tf.add_to_collection('losses', cross_entropy_mean)
cross_entropy = tf.add_n(tf.get_collection('losses'), name='total_loss')
l_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
prediction = tf.argmax(y_conv,1)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
  
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

learn_rate = 2e-3
phist = .5
for i in range(100):
	batch = my_next_batch(50)
	if i%100 == 0:
         train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], 
                                                   keep_prob: 1.0, name: batch[2], l_rate: learn_rate})
         print('step %d, training accuracy %g'%(i, train_accuracy))
         if np.abs(phist - train_accuracy) / phist < .1 :
             learn_rate /= 1.0
         if i % 2000 == 0:
             if learn_rate >= 1e-6:
                 learn_rate /= 2.
         phist = train_accuracy	
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1., l_rate: learn_rate}) # changed keep_prob to 1.
	
save_path = saver.save(sess, 'my-model2') 
print ('Model saved in file: ', save_path) 

eqims = geteqims(trainpath) # tuple: (ndarray, path) for images of equations
#ims comps is a list of 2-element tuples: ((list of ndarrays for components, list of corresponding bounding box coordinates), equationpath)
imscomps = [(connectedcomps(i[0]), i[1]) for i in eqims] # 

# uses variables defined outside function: imscomps
def formatcomps():
	testdata = [] 
	for eq in imscomps: # components and path for a particular equation
		# eq[0] is a tuple: (list of ndarrays for components, list of corresponding bounding box coordinates)
		numcomps = len(eq[0][0])
		for i in range(len(eq[0][0])):
			testdata += [(np.resize(eq[0][0][i], 32*32), eq[0][1][i], eq[1], numcomps)]
	return testdata 

testdata = formatcomps()

def structuretestdata(): 
	""" ** Description of method ***
	
	Returns
	-------
	(array, array, array)
		x - 28x28 tensor for image pixels (one single component)
		y - bounding box coordinates for x (in equation)
		z - holds the image path for the original equation
		num - number of components for the equation in z
	"""
	size = len(testdata)
	x = np.zeros((size, 32*32), dtype=np.float32) # important to specify dtype=np.float32, otherwise UnicodeDecodeError
	y = np.empty((size, 4), dtype=np.int32) # holds bounding box coordinates
	z = np.empty((size, 1), dtype='<U150') # this is for image paths. row is for each image and column is 1 because it's just one string
	# t = np.zeros((size, 28*28)) # (28, 28); this can be used with scm.imsave if you want to save the image for the component as I used to do.
	num = np.empty((size, 1), dtype=np.int32)
	for j in range(size):
		x[j] = np.asarray(testdata[j][0], dtype=np.float32)
		y[j] = np.asarray(testdata[j][1], dtype=np.int32)
		z[j] = np.asarray(testdata[j][2])
		num[j] = np.asarray(testdata[j][3])
	return x, y, z, num

(testims, testbb, testpaths, num) = structuretestdata()
pred = prediction.eval(feed_dict={x: testims, keep_prob: 1.0}) 
bboxes = identitybox.eval(feed_dict={box: testbb})
paths = identityname.eval(feed_dict={name: testpaths})
paths = [getlocalpath(str(p[0], 'utf-8')) for p in paths]
numcomps = identitynum.eval(feed_dict={n: num})

f = open('predictions.txt', 'w')
prev = paths[0]
for i in range(len(paths)):
	p = paths[i]
	if p != prev or i == 0:
		f.write(p + '\t' + str(numcomps[i][0]) + '\t\n')
	f.write(str(SymPred(labellst[pred[i]], bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2])) + '\n')
	prev = p
f.close()
