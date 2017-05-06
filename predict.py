#add your imports here
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import random
import tensorflow as tf

import sys
#import glob
from PIL import Image, ImageFilter
import skimage.measure as skm
import scipy.misc as scm
import numpy as np
import skimage.morphology as morphology
import tensorflow as tf

trainpath = sys.argv[-1]  # # path for the folder containg the training images i.e. the path for 'annotated'
# tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables


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
	paths = glob(folder+'/*.png')
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

sess.run(tf.global_variables_initializer())

"""
add whatever you think it's essential here
"""
class SymPred():
	def __init__(self,prediction, x1, y1, x2, y2):
		"""
		<x1,y1> <x2,y2> is the top-left and bottom-right coordinates for the bounding box
		(x1,y1)
			   .--------
			   |	   	|
			   |	   	|
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

class ImgPred():
	def __init__(self,image_name,sym_pred_list,latex = 'LATEX_REPR'):
		"""
		sym_pred_list is list of SymPred
		latex is the latex representation of the equation 
		"""
		self.image_name = image_name
		self.latex = latex
		self.sym_pred_list = sym_pred_list
	def __str__(self):
		res = self.image_name + '\t' + str(len(self.sym_pred_list)) + '\t' + self.latex + '\n'
		for sym_pred in self.sym_pred_list:
			res += str(sym_pred) + '\n'
		return res
	
# Define TensorFlow Session
sess = tf.Session()

# Import saved trained model
tf.train.Saver().restore(sess, "./my-model")
print("Model restored.")



def predict(image_path):
	
	"""
	Add your code here
	"""
	
	"""
	#Don't forget to store your prediction into ImgPred
	img_prediction = ImgPred(...)
	"""
	img_transf = transform(scm.imread(image_path))
	img_transf = np.transpose(img_transf.reshape(784,-1))
	img_prediction = prediction.eval(feed_dict={x: img_transf, keep_prob: 1.0}) 
	return img_prediction[0]

if __name__ == '__main__':
	image_folder_path = argv[1]
	isWindows_flag = False
	if len(argv) == 3:
		isWindows_flag = True
	if isWindows_flag:
		image_paths = glob(image_folder_path + '\\*png')
	else:
		image_paths = glob(image_folder_path + '/*png')
	results = []
	for image_path in image_paths:
		impred = predict(image_path)
		results.append(impred)

	with open('predictions.txt','w') as fout:
		for res in results:
			fout.write(str(res))