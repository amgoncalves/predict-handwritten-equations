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

import sys
from tensorflow.examples.tutorials.mnist import input_data 
import glob
from PIL import Image, ImageFilter
import PIL.ImageOps
import scipy.misc as misc
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf
from skimage.transform import resize,warp,AffineTransform
import random

trainpath1 = sys.argv[1] #  path for the folder containg the training images i.e. the path for 'annotated'
trainpath2 = sys.argv[2] # path for folder of data from https://www.kaggle.com/xainano/handwrittenmathsymbols

invert_flag2 = False
tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables
imrows = 32 
imcols = 32
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
  
def input_wrapper(image):
	sx,sy = image.shape
	diff = np.abs(sx-sy)

	sx,sy = image.shape
	image = np.pad(image,((sx//8,sx//8),(sy//8,sy//8)),'constant')
	if sx > sy:
		image = np.pad(image,((0,0),(diff//2,diff//2)),'constant')
	else:
		image = np.pad(image,((diff//2,diff//2),(0,0)),'constant')
	
	image = morphology.dilation(image, morphology.disk(max(sx,sy)/32))
	image = misc.imresize(image,(32,32))
	if np.max(image) > 1:
		image = image/255.
	return image

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
	im = (im.crop((left,upper,right,lower)))
	return np.array(im.getdata()).reshape((im.size[1], im.size[0]))
	
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
            for impath in e:
                label = getlabel(impath)
                if label not in ('cos', 'sin', 'tan'):
                    sypaths += [impath]
    return sypaths

def geteqpaths(folder):
	d = getdict(folder)
	return list(d.keys())
		 
# Get the images of the symbols. These will be used as training data
# list of tuples: (ndarray length imrows*imcols of image, imagepath)
def getsyims(folder):
	return [(input_wrapper(cropim(misc.imread(impath))), impath) for impath in getsypaths(folder)]
			
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
	
# Add the corresponding label to each tuple for the argument trainims, which is the result of getsyims(trainpath1)
def addlabel(trainims):
	""" Add the corresponding label to each tuple for the argument trainims, which is the result of getsyims(trainpath1).
	
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
	return (ims, paths, labels)		   

folders2 = glob.glob(trainpath2+'/*')
# These are math symbols that I've confirmed there are folders for in the Kaggle data set.
# Not all the math symbols e.g. frac are here. Also not that div in data also includes / 
#mathsymbols= ['(',')','+','-', '=', 'cos', 'delta', 'div', 
#                  'dots', 'mul', 'pi', 'pm', 'sin', 'sqrt', 'tan'] # size 127,361 data set   
mathsymbols= ['(',')','+','-', '=', 'delta', 'div', 
                  'dots', 'mul', 'pi', 'pm', 'sqrt'] # size 127,361 data set                   
letters = ['A', 'a', 'b', 'c', 'd', 'f', 'h', 'i', 'k',
           'm', 'n', 'o', 'p', 's', 't', 'x', 'y']
numbers = ['0', '1', '2', '3', '4', '6'] 
symbols = mathsymbols + letters + numbers # doesn't include bar and frac; seems like I've inverted all images
### Letters: uppercase and lowercase have to be separated in the folders since they're all mixed together
### the issue is that some file names have e.g. a____, A____, exp____ so the exp____ ones have to be deleted/filtered out

"""
trainfolders2 = []
for f in folders2:
    localpath = getlocalpath(f)
    if localpath.lower() in symbols or localpath.upper() in symbols:
        trainfolders2 += [f]

def invert_train_data2():
    for f in trainfolders2:
        impaths = glob.glob(f+'/*.jpg')
        for path in impaths:
            im = (Image.open(path)).convert('L') # convert is necessary so it's converted from RGB to grayscale
            im = (PIL.ImageOps.invert(im)).filter(ImageFilter.SHARPEN)
            im.save(path[0:(len(path)-3)] + 'png') # save as png      

def get_train_data2():
    ims, labels = [], []
    for f in trainfolders2:
        label = getlocalpath(f)
        impaths = glob.glob(f+'/*.png')
        i = 0
        for path in impaths:
            localpath = getlocalpath(path)
            # if localpath[0] == 'e', can't determine if upper or lower case for letters for exp___.png files
            # So that we have similar number of symbols for each one, continue even for symbols that aren't letter.
            if localpath[0] == 'e':
                continue
            localpath = localpath[0:(len(localpath)-4)] # remove .png
            if (label.lower() in letters or label.upper() in letters) and localpath[0] in letters:
                label = localpath[0]
            if label in symbols:
                ims.append(input_wrapper(cropim(misc.imread(path))))
                labels.append(label)
            i += 1
    return (ims, labels)
"""
  
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
   
syims = addlabel(getsyims(trainpath1)) # symbol (not equation) images; result is list of 3-element tuples: 
(trainims, _, labels) = unpack(syims)

"""
if invert_flag2:
    invert_train_data2()
(trainims2, labels2) = get_train_data2()
"""

label_lst = list(set(labels)) 
label_lst.sort() # sorted list of unique labels
onehotdict = oneHot(label_lst)

#full_trainims = trainims + trainims2 
#full_labels = labels + labels2

#full_trainims = trainims2 
#full_labels = labels2

full_trainims = trainims
full_labels = labels


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
def next_mnist_batch(batch_size=1):
    is_valid_digit = lambda onehot: any((onehot[0], onehot[1], onehot[2], 
                                         onehot[3], onehot[4], onehot[6]))
    batch = []
    while len(batch) < batch_size:
        mnist_batch = mnist.train.next_batch(1)
        if is_valid_digit(mnist_batch[1][0]):
            batch.append(mnist_batch)
    return batch 

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
    # randomly pick batch_size number of elements from trainims
    n = len(full_trainims)
    train_size = n
    indices = [np.random.randint(0, train_size) for j in range(batch_size)]
    numlabels = len(label_lst)
    batch_x = np.zeros((batch_size, imrows*imcols))
    batch_y = np.zeros((batch_size, numlabels)) # rows = batch_size and cols = # of unique symbols
    for j in range(batch_size):
        k = indices[j]
        if k < n:
            batch_x[j] = np.asarray(np.reshape(image_deformation(full_trainims[k]), imrows*imcols))
            batch_y[j] = np.asarray(onehotdict[full_labels[k]])
        else: # currently not being called
            mnist_batch = next_mnist_batch()[0]
            imdata = mnist_batch[0][0]
            if imrows != 28 or imcols != 28:
                imdata = misc.imresize(np.reshape(imdata, (28, 28)), (imrows, imcols), 'bicubic')
            else:
                imdata = np.reshape(imdata, (imrows, imcols))
            batch_x[j] = np.asarray(np.reshape(image_deformation(imdata), imrows*imcols))
            onehotindex = mnist_batch[1][0].argmax() # argmax returns index of maximum value of one hot tensor (between 0 and 9)
            batch_y[j] = np.asarray(onehotdict[str(onehotindex)])
    return batch_x, batch_y
    

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
   
x = tf.placeholder(tf.float32, shape=[None, imrows*imcols])
y_ = tf.placeholder(tf.float32, shape=[None, len(label_lst)]) # len(label(lst)) is the number of unique labels

padding = 'SAME'

"""
#CONVOLUTION LAYER 1
"""

W_conv1 = weight_variable([5, 5, 1, 32])
#W_conv1 = weight_variable([7, 7, 1, 32])
#W_conv1 = weight_variable([3, 3, 1, 8])
b_conv1 = bias_variable([32])
#b_conv1 = bias_variable([8])

x_image = tf.reshape(x, [-1,imrows,imcols,1]) 

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1) 
#tmp_1, _, _ = batch_norm_layer(conv2d(x_image, W_conv1))
#h_conv1 = tf.nn.relu(tmp_1)
h_pool1 = max_pool_2x2(h_conv1) 

"""
#CONVOLUTION LAYER 2
"""
W_conv2 = weight_variable([5, 5, 32, 64]) 
#W_conv2 = weight_variable([7, 7, 32, 64]) 
#W_conv2 = weight_variable([3, 3, 8, 16]) 
b_conv2 = bias_variable([64])
#b_conv2 = bias_variable([16])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
#tmp_2, _, _ = batch_norm_layer(conv2d(h_pool1, W_conv2))
#h_conv2 = tf.nn.relu(tmp_2)
h_pool2 = max_pool_2x2(h_conv2)

"""
#DENSELY CONVOLUTED LAYER
"""	   
W_fc1 = weight_variable([imrows//4 * imcols//4 * 64, 1024])
b_fc1 = bias_variable([1024]) 
#W_fc1 = weight_variable([imrows//4 * imcols//4 * 16, 256])
#b_fc1 = bias_variable([256]) 
	
h_pool2_flat = tf.reshape(h_pool2, [-1, imrows//4 * imcols//4 * 64])
#h_pool2_flat = tf.reshape(h_pool2, [-1, imrows//4 * imcols//4 * 16])
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
#W_fc2 = weight_variable([256, len(label_lst)])
b_fc2 = bias_variable([len(label_lst)])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
"""
#TRAIN & EVALUATE
"""
cross_entropy = tf.reduce_mean(
	 tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
l_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
prediction = tf.argmax(y_conv,1) 
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.global_variables_initializer())
 
#learn_rate = 8e-4
learn_rate = 6e-4
phist = .5
for i in range(20000):
    batch = my_next_batch(40)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], 
                                                   keep_prob: 1.0, l_rate: learn_rate})
        print('step %d, training accuracy %g'%(i, train_accuracy))
        if np.abs(phist - train_accuracy) / phist < .1:
            learn_rate /= 1.0
        if i % 2000 == 0:
            if learn_rate >= 1e-6:
                learn_rate /= 2.
        phist = train_accuracy	
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5, l_rate: learn_rate})
saver = tf.train.Saver()
save_path = saver.save(sess, 'my-model') 
print ('Model saved in file: ', save_path) 