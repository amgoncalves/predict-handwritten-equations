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
To-do's:
    1. Remove every 5th in the get train 3 fn
    
"""

"""
trainims: 3,452, trainims3: 18,349 (every 5th), fulltrainims: 21,801
a = [[e == labels3[i] for i in range(len(labels3))] for e in symbols]
b = [sum(lst) for lst in a]
b
Out[95]: 
[930,                 
 929,
 1642,
 2251,
 848,
 7,
 44,
 31,
 182,
 125,
 41,
 494,
 68,
 590,
 464,
 310,
 262,
 193,
 74,
 281,
 157,
 133,
 579,
 23,
 136,
 71,
 172,
 1538,
 517,
 445,
 1730,
 1715,
 702,
 467,
 198]
sum(b)
Out[96]: 18349
"""
import sys
from tensorflow.examples.tutorials.mnist import input_data 
import glob
from PIL import Image, ImageFilter
import PIL.ImageOps
import scipy.misc as scm
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf
from skimage.transform import resize,warp,AffineTransform
import random

trainpath1 = sys.argv[1] #  path for the folder containg the training images i.e. the path for 'annotated'
trainpath2 = sys.argv[2] # path for folder of data from http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
trainpath3 = sys.argv[3] # path for folder of data from https://www.kaggle.com/xainano/handwrittenmathsymbols

invert_flag2 = False
invert_flag3 = False
tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables
imrows = 28 # can't get it to work with imrows = 32, imcols = 32 for some reason
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
    if im.max() > 1: # only divide by .max() if it's greater than 1 meaning data needs to be normalized.
        return im/im.max() # MNIST data says 0 means white and 255 means black. MNIST images are normalized between 0 and 1. 
    return im

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


"""
# This transform is used only if we do not use image_deformation
def transform(im):
	return normalize(np.reshape(morphology.dilation(scm.imresize(fullpadim(im), (imrows, imcols), 'bicubic')), imrows*imcols))
"""
# Removed np.reshape
def transform(im):
	return normalize(morphology.dilation(scm.imresize(fullpadim(im), (imrows, imcols), 'bicubic')))
 
# Alternate way of dealing with training data: don't pad or dilate. Leads to very inaccurate results.
def transform2(im):
	return normalize(scm.imresize(im, (imrows, imcols), 'bicubic'))
 
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
	#return (np.asarray(ims), np.asarray(paths), np.asarray(labels)) # currently seems unnecessary based on what I'm doing in my_next_batch
	return (ims, paths, labels)		   

"""
trainfolders2 = glob.glob(trainpath2+'/*')
is_valid_folder2 = lambda s: any((s[-1] == 'A', s[-1] == 'a', s[-1] == 'b', s[-1] == 'c',
                                     s[-1] == 'd', s[-1] == 'f', s[-1] == 'i', s[-1] == 'k',
                                     s[-1] == 'm', s[-1] == 'n', s[-1] == 'o', s[-1] == 'p',
                                     s[-1] == 's', s[-1] == 't', s[-1] == 'x', s[-1] == 'y', s[-1] == 'h'))
trainfolders2 = [f for f in trainfolders2 if is_valid_folder2(f)]
 
# uses outside variable folders
def invert_train_data2():
    for f in trainfolders2:
        impaths = glob.glob(f+'/*.png')
        for path in impaths:
            im = (Image.open(path)).convert('L') # convert is necessary so it's converted from RGB to grayscale
            im = (PIL.ImageOps.invert(im)).filter(ImageFilter.SHARPEN)
            im.save(path)
            
# Assumes images in traindata2 have already been inverted (black & white) by invert_train_data2
def get_train_data2(): # size 935 data set
    ims, labels = [], []
    for f in trainfolders2:
        label = f[-1]
        impaths = glob.glob(f+'/*.png')
        for path in impaths:
            ims.append(transform(scm.imread(path)))
            labels.append(label)
    return (ims, labels)
  
"""

folders3 = glob.glob(trainpath3+'/*')
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

trainfolders3 = []
for f in folders3:
    localpath = getlocalpath(f)
    if localpath.lower() in symbols or localpath.upper() in symbols:
        trainfolders3 += [f]

                    # remove cos, sin, tan?
#trainfolders3 = [f for f in folders3 if 
 #                (getlocalpath(f).lower() in symbols or getlocalpath(f).upper() in symbols)]
# If I decide to just use all the symbols from the data set then just start from scratch and use label_lst as valid_folders3
# Just make sure the symbols match the folder names. Could also just use the symbols in valid_folders3_extra since I already
# have MNIST data. It depends, but I think due to the size of the data set

### trainfolders3 is still missing = for some reason
### I'M GOING TO HAVE TO INVERT STUFF AGAIN 'CAUSE MY TRAINFOLDERS3 IS WRONG ###
def invert_train_data3():
    for f in trainfolders3:
        impaths = glob.glob(f+'/*.jpg')
        for path in impaths:
            im = (Image.open(path)).convert('L') # convert is necessary so it's converted from RGB to grayscale
            im = (PIL.ImageOps.invert(im)).filter(ImageFilter.SHARPEN)
            im.save(path[0:(len(path)-3)] + 'png') # save as png      

def get_train_data3():
    ims, labels = [], []
    for f in trainfolders3:
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
            #if (label.lower() in letters or label.upper() in letters) and localpath[0] in letters:
                #label = localpath[0] 
            if (label.lower() in letters or label.upper() in letters) and localpath[0] in letters:
                label = localpath[0]
            #if i%5 == 0: # EVERY 5th for now. 
            if label in symbols:
                ims.append(transform(scm.imread(path)))
                labels.append(label)
            #ims.append(transform(scm.imread(path))) 
            #labels.append(label)
            i += 1
    return (ims, labels)
    
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
   
syims = addlabel(getsyims(trainpath1)) # symbol (not equation) images; result is list of 3-element tuples: 
(trainims, _, labels) = unpack(syims)

"""
if invert_flag2:
    invert_train_data2()
(trainims2, labels2) = get_train_data2()
"""

if invert_flag3:
    invert_train_data3()
(trainims3, labels3) = get_train_data3()

label_lst = list(set(labels)) 
label_lst.sort() # sorted list of unique labels
onehotdict = oneHot(label_lst)

#full_trainims = trainims + trainims2 + trainims3 
#full_labels = labels + labels2 + labels3

full_trainims = trainims + trainims3 
full_labels = labels + labels3

"""
After some commands in a python console, I determined that batch = mnist.train.next_batch(1) would return a
tuple of arrays. batch[0] would be an array holding a (784, ) shape array i.e. batch[0][0].shape 
is (784, ). batch[1] would be an array holding a (10, ) array i.e. batch[1][0].shape is (10, 1)
Clearly the batch[0][0] is the array for the image and batch[1][0] is the array for the one hot tensor
We can use the latter array to filter out numbers that we don't need using a UDF that builds a batch
of our desired size (for now, batch size 25) as we read mnist batches of size one.
Perhaps use lambda function to filter out. 

However, since technically we're not using MNIST exclusively, give it a default argument set to one for batch size
so that my_next_batch calls it randomly when needing to build a batch size of 25.

mnist.train.images is a [55000, 784] tensor.

"""

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
    #train_size = n + 6*55000//10 #  3480 + ... + 33000
    train_size = n
    # MNIST has 55,000 training images. Assume 5,500 images for each digits.
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
            print('MNIST')
            mnist_batch = next_mnist_batch()[0]
            imdata = mnist_batch[0][0]
            if imrows != 28 or imcols != 28:
                imdata = scm.imresize(np.reshape(imdata, (28, 28)), (imrows, imcols), 'bicubic')
            else:
                imdata = np.reshape(imdata, (imrows, imcols))
            batch_x[j] = np.asarray(np.reshape(image_deformation(imdata), imrows*imcols))
            onehotindex = mnist_batch[1][0].argmax() # argmax returns index of maximum value of one hot tensor (between 0 and 9)
            batch_y[j] = np.asarray(onehotdict[str(onehotindex)])
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

for i in range(20000): # then try 15000
    batch = my_next_batch(20)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
                                                  x: batch[0], y_: batch[1], keep_prob: 1.0})
        print('step %d, training accuracy %g'%(i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

saver = tf.train.Saver()
save_path = saver.save(sess, 'my-model') 
print ('Model saved in file: ', save_path) 
