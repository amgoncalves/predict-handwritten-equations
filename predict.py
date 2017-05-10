#add your imports here
from sys import argv
from glob import glob
from scipy import misc
import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import skimage.measure as skm
import skimage.morphology as morphology
import tensorflow as tf

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
imrows = 32
imcols = 32

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
    comps = skm.regionprops(skm.label(im > 0, connectivity=1)) 
    bbcoords = []
    newcomps = []
    for i in range(len(comps)):
        if comps[i].area < 50:
            continue
        bbcoords += [comps[i].bbox]
        newcomps += [input_wrapper(cropim(np.asarray(comps[i].image, dtype=np.float32)))]
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
y_ = tf.placeholder(tf.float32, shape=[None, len(label_lst)]) # len(label_lst)) is the number of unique labels

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

# Define TensorFlow Session
#sess = tf.Session()


# Import saved trained model
tf.train.Saver().restore(sess, "./my-model")
print("Model restored.")

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
l_rate = tf.placeholder(tf.float32)
train_step = tf.train.AdamOptimizer(l_rate).minimize(cross_entropy)
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
     (comps, bboxes) = connectedcomps(misc.imread(image_path))
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
                