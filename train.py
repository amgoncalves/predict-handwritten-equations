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
    1. Consider changing imrows and imcols to 32, 32
    2. Consider using TA's input wrapper instead of padim and fullpadim
    3. How does it work without dilation? How about Sharpen (didn't work last time however)?
    4. More data: MNIST,  http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    5. More steps: 15000-20000
    6. More complex NN. Could the batch norm part have thrown off the complex NN results during my prior attempt?
    7. Consider batch normalization again since we're using a more complex NN.
    8. Improve partitioning of components:
    e.g. SKMBT_36317040717260_eq12.png - only 6 components for some reason? Does this has to do with comps[i].area < 50 in connectedcomps?
    e.g. SKMBT_36317040717260_eq3.png - Only 2 rather than 3 comps. It looks like the 3 is not being counted as a component. Perhaps 
    I need to decrease the threshold area, or normalize the images in some way so that I can have a threshold that can be justified 
    to apply to all images.
"""
import sys
import glob
from PIL import Image, ImageFilter
import scipy.misc as scm
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf
from skimage.transform import resize,warp,AffineTransform

trainpath = sys.argv[-1]  # # path for the folder containg the training images i.e. the path for 'annotated'
tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables
imrows = 28
imcols = 28
np.random.seed(10)

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

"""
# This transform is used only if we do not use image_deformation
def transform(im):
	return normalize(np.reshape(morphology.dilation(scm.imresize(fullpadim(im), (imrows, imcols), 'bicubic')), imrows*imcols))
"""
# Removed np.reshape
def transform(im):
	return normalize(morphology.dilation(scm.imresize(fullpadim(im), (imrows, imcols), 'bicubic')))
 
def geteqims(folder):
	return [(scm.imread(impath), impath) for impath in geteqpaths(folder)]
		 
# Get the images of the symbols. These will be used as training data
# list of tuples: (ndarray length imrows*imcols of image, imagepath)
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

# args: lst - sorted list of unique labels e.g. label_lst = list(set(labels)).sorted()
# returns dictionary of onehot lists for each label
def oneHot(lst):
	""" *** Description ***
	
	Parameters
	----------
	lst : list
		Sorted list of unique labels. e.g. label_lst = list(set(labels)).sorted()
		
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

"""
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
"""
        
syims = addlabel(getsyims(trainpath)) # symbol (not equation) images; result is list of 3-element tuples: 

(trainims, _, labels) = unpack(syims)
label_lst = list(set(labels)) 
label_lst.sort() # sorted list of unique labels
onehotdict = oneHot(label_lst)

# uses variables defined outside of this function: trainims, label_lst
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
     ### add image deformation ###
	size = len(trainims)
	indices = [np.random.randint(0, size) for j in range(batch_size)]
	numlabels = len(label_lst)
	batch_x = np.zeros((batch_size, imrows*imcols))
	batch_y = np.zeros((batch_size, numlabels)) # rows = batch_size and cols = # of unique symbols
	for j in range(batch_size):
		k = indices[j]
		batch_x[j] = np.asarray(np.reshape(image_deformation(trainims[k]), imrows*imcols)) 
		batch_y[j] = np.asarray(onehotdict[labels[k]])
	return batch_x, batch_y
		
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
   
x = tf.placeholder(tf.float32, shape=[None, imrows*imcols])
y_ = tf.placeholder(tf.float32, shape=[None, len(label_lst)]) # len(label(lst)) is the number of unique labels

"""
#CONVOLUTION LAYER 1
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,imrows,imcols,1]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
#tmp_1, _, _ = batch_norm_layer(conv2d(x_image, W_conv1))
#h_conv1 = tf.nn.relu(tmp_1)
h_pool1 = max_pool_2x2(h_conv1) 

"""
#CONVOLUTION LAYER 2
"""
W_conv2 = weight_variable([5, 5, 32, 64]) 
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#tmp_2, _, _ = batch_norm_layer(conv2d(h_pool1, W_conv2))
#h_conv2 = tf.nn.relu(tmp_2)
h_pool2 = max_pool_2x2(h_conv2)

"""
#DENSELY CONVOLUTED LAYER
"""	   
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024]) 
	
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1) # batch_norm_layer is not applied in this layer
	
"""
#DROPOUT
"""
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
	
"""
#READOUT LAYER
"""
W_fc2 = weight_variable([1024, len(label_lst)])
b_fc2 = bias_variable([len(label_lst)])

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

sess.run(tf.global_variables_initializer())

for i in range(10000): # then try 20000
	batch = my_next_batch(25)
	if i%100 == 0:
		train_accuracy = accuracy.eval(feed_dict={
												  x: batch[0], y_: batch[1], keep_prob: 1.0})
		print('step %d, training accuracy %g'%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

saver = tf.train.Saver()
save_path = saver.save(sess, 'my-model') 
print ('Model saved in file: ', save_path) 