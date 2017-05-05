from __future__ import division
from __future__ import print_function
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:06:17 2017

@author: paulcabrera
"""

"""
Comments
	- I might have to check why there's only 387 elements in eqims and imscomps. Pretty sure
	there should be 700+? Or I might be remembering incorrectly.
"""

import sys
import glob
from PIL import Image, ImageFilter
import skimage.measure as skm
import scipy.misc as scm
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf

trainpath = sys.argv[-1]  # # path for the folder containg the training images i.e. the path for 'annotated'
# tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables

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
									(28, 28), 'bicubic')))]
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

def transform(im):
	return normalize(np.reshape(morphology.dilation(scm.imresize(fullpadim(im), (28, 28), 'bicubic')), 28*28))

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

# uses variables defined outside of this function: trainims, trainpaths, labellst
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
	batch_x = np.zeros((batch_size, 28*28))
	batch_y = np.zeros((batch_size, numlabels)) # rows = batch_size and cols = # of unique symbols
	batch_z = np.empty((batch_size, 1), dtype='<U150') # this is for image paths. row is for each image and column is 1 because it's just one string
	for j in range(batch_size):
		k = indices[j]
		batch_x[j] = np.asarray(trainims[k]) 
		batch_y[j] = np.asarray(onehotdict[labels[k]])
		batch_z[j] = np.asarray(trainpaths[k])
	return batch_x, batch_y, batch_z
		
def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)
	  
def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)
	 
def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
	  
def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
							strides=[1, 2, 2, 1], padding='SAME')
	  
sess = tf.InteractiveSession()
   
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, len(labellst)]) # len(label(lst)) is the number of unique labels
box = tf.placeholder(tf.int32, shape=[None, 4])
name = tf.placeholder(tf.string, shape=[None, 1])
n = tf.placeholder(tf.int32, shape=[None, 1])
	
"""
#CONVOLUTION LAYER 1
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,28,28,1]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
h_pool1 = max_pool_2x2(h_conv1) 

"""
#CONVOLUTION LAYER 2
"""
W_conv2 = weight_variable([5, 5, 32, 64]) 
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

"""
#DENSELY CONVOLUTED LAYER
"""	   
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
	
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
	
"""
#DROPOUT
"""
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
"""
#READOUT LAYER
"""
W_fc2 = weight_variable([1024, len(labellst)])
b_fc2 = bias_variable([len(labellst)])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
"""
#TRAIN & EVALUATE
"""
cross_entropy = tf.reduce_mean(
	 tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
prediction = tf.argmax(y_conv,1) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
identitybox = tf.identity(box)
identityname = tf.identity(name)
identitynum = tf.identity(n)

saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

for i in range(10000): # then try 20000
	batch = my_next_batch()
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
												  x: batch[0], y_: batch[1], keep_prob: 1.0, name: batch[2]})
		print('step %d, training accuracy %g'%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	
save_path = saver.save(sess, 'my-model') 
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
			testdata += [(np.resize(eq[0][0][i], 28*28), eq[0][1][i], eq[1], numcomps)]
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
	x = np.zeros((size, 28*28), dtype=np.float32) # important to specify dtype=np.float32, otherwise UnicodeDecodeError
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
paths = [getlocalpath(str(p[0]).encode('utf-8')) for p in paths]
numcomps = identitynum.eval(feed_dict={n: num})

g = open('test-predictions.txt', 'w')	
for i in range(len(pred)):
	g.write(str(paths[i]) + "\t" + str(labellst[pred[i]]) + "\n")
g.close()

f = open('predictions.txt', 'w')
prev = paths[0]
for i in range(len(paths)):
	p = paths[i]
	if p != prev or i == 0:
		f.write(p + '\t' + str(numcomps[i][0]) + '\t\n')
	f.write(str(SymPred(labellst[pred[i]], bboxes[i][1], bboxes[i][0], bboxes[i][3], bboxes[i][2])) + '\n')
	prev = p
f.close()
