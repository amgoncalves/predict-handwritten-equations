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

"""
Comments:
    Debugging - I made sure that the one hot tensors have the 1 in the right place given label_lst in predict.py
    [onehotdict[sym].index(1) for sym in label_lst] shows that the one hot tensors are correct.
"""

tf.reset_default_graph() # http://stackoverflow.com/questions/41400391/tensorflow-saving-and-resoring-session-multiple-variables
"""
label_lst = ['(',')','+','-','0','1','2','3','4','6',
             '=', 'A', 'a', 'b', 'bar', 'c', 'cos', 'd',
             'delta', 'div', 'dots', 'f', 'frac', 'h', 'i', 'k',
             'm', 'mul', 'n', 'o', 'p', 'pi', 'pm', 's', 'sin',
             'sqrt', 't', 'tan', 'x', 'y'] # this corresponds to labellst that was created in the training script
"""
label_lst = ['(',')','+','-','0','1','2','3','4','6',
             '=', 'A', 'a', 'b', 'bar', 'c', 'd',
             'delta', 'div', 'dots', 'f', 'frac', 'h', 'i', 'k',
             'm', 'mul', 'n', 'o', 'p', 'pi', 'pm', 's',
             'sqrt', 't', 'x', 'y'] # this corresponds to labellst that was created in the training script
label_lst.sort()             
imrows = 28
imcols = 28
#imshape = (4000, 4000)      


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
    if im.max() > 1:
        return im/im.max() # MNIST data says 0 means white and 255 means black. MNIST images are normalized between 0 and 1. 
    return im

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
									(imrows, imcols), 'bicubic')))]
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

def transform(im):
	return normalize(np.reshape(morphology.dilation(scm.imresize(fullpadim(im), (imrows, imcols), 'bicubic')), imrows*imcols))

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
y_ = tf.placeholder(tf.float32, shape=[None, len(label_lst)]) # len(label_lst)) is the number of unique labels

"""
#CONVOLUTION LAYER 1
"""
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x, [-1,imrows,imcols,1]) 

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
W_fc2 = weight_variable([1024, len(label_lst)])
b_fc2 = bias_variable([len(label_lst)])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
	
"""
#TRAIN & EVALUATE
"""

# Define TensorFlow Session
#sess = tf.Session()


# Import saved trained model
tf.train.Saver().restore(sess, "./my-model")
print("Model restored.")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

prediction = tf.argmax(y_conv,1)

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
	




def predict(image_path):
	
     """
     Add your code here
     """
	
     """
     #Don't forget to store your prediction into ImgPred
     img_prediction = ImgPred(...)
     """
     #split_path  = image_path.split("/")
     #file_name = split_path[len(split_path)-1]
     (comps, bboxes) = connectedcomps(scm.imread(image_path))
     numcomps = len(comps)
     results =[]
     i = 0
     for i in range(numcomps):
         img, box = comps[i], bboxes[i]
         img_transf = np.transpose(img.reshape(imrows*imcols,-1))
         img_prediction = prediction.eval(feed_dict={x: img_transf, keep_prob: 1.0}) 
         results.append((img_prediction, box, getlocalpath(image_path), numcomps))
     return results

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
        for eqres in results: # results for a particular equation
            fout.write(eqres[0][2] + '\t' + str(eqres[0][3]) + '\t\n') # project specifies extra tab at end
            for comp in eqres:
                pred, bbox = comp[0], comp[1]
                fout.write(str(SymPred(label_lst[pred[0]], bbox[1], bbox[0], bbox[3], bbox[2])) + '\n')
                