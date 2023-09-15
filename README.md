# Handwritten_Digit_Recognition
Handwritten Digit Recognition using TensorFlow and MNIST dataset

## Files to submit:
A report in PDF format. For each model that you have tested, please describe the model, the graph representation captured in TensorBoard, the accuracy you achieved and brief analysis, time you spent to complete this task, and interesting problems that you've met;
a1a.py Download a1a.py Python source code for logistic regression.  You need to add your code to complete the functionality;
a1b.py  Additional python source code for other models like DNN (deep neural network), CNN (convolutional neural network), and so on. You need to create the file.
utils.py Download utils.py Python source code for utility functions that will be used by a1a.py.  You do not need to modify this file.

## Learning Goal:
How to prepare image data for training?
batching
How to use a high-level framework (TensorFlow) to create a neural model?
simple logistic regression
fully-connected and convolution layers
stacking layers
How to train the model?
typical training loop
model evaluation

## Platform: TensorFlow
Please follow the official instruction to install TensorFlow here Links to an external site.. You need to have Python 3.7+ gets installed to finish this assignment. It is recommended to use Conda Links to an external site. to manage your environment. 

A reference list of dependencies:
~~~
ipdb==0.13.9
lxml==4.7.1
matplotlib==3.4.1
numpy==1.19.2
Pillow==9.0.0
scikit-learn==1.0.2
scipy==1.7.3
tensorflow==2.5.0
xlrd==2.0.1
~~~
We can encode the above into a requirements.txt. You can simply run the following commands to create an environment and install the dependencies.
~~~
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
~~~
Then you can try to run logreg_example.py (you also need to download another file input_data.py before running it) using the following command to verify whether tensorflow is correctly installed.
~~~
python logreg_example.py
~~~

## Dataset: MNIST
The MNIST (Mixed National Institute of Standards and Technology database) is one of the most popular databases used for training various image processing systems. It is a database of handwritten digits.
Each image is 28 x 28 pixels. You can flatten each image to be a 1-d tensor of size 784. Each comes with a label from 0 to 9. For example, images on the first row are labeled as 0, the second as 1, and so on. The dataset is hosted on Yann Lecun’s website http://yann.lecun.com/exdb/mnist/

## Data Loading Approach 1.
TensorFlow has a function call that lets you load the MNIST dataset and divide it into train set, and test set.
~~~
import tensorflow as tf
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
Above statement returns 60,000 data points as training data,  10,000 data points as testing data.
~~~
### One-hot encoding
In digital circuits, one-hot refers to a group of bits among which the legal combinations of values are only those with a single high (1) bit and all the others low (0).

In this case, one-hot encoding means that if the output of the image is the digit 7, then the output will be encoded as a vector of 10 elements with all elements being 0, except for the element at index 7 which is 1.
~~~
y_train = tf.one_hot(y_train, depth=10)
~~~
Above statement converts each digit to its One-hot encoding representation format.

## Data Loading Approach 2.
You can use the provided utils.py, which implemented functions

downloading and parsing MNIST data into numpy arrays in the file utils.py. All you need to do in your program is: 
~~~
import utils
mnist_folder = 'data/mnist'
utils.download_mnist(mnist_folder)
train, val, test = utils.read_mnist(mnist_folder, flatten=True)
~~~
We need choose flatten = True for logistics regression and DNN, because we want each image to be flattened into a 1-d tensor. Each of train, val, and test in this case is a tuple of NumPy arrays, the first is a NumPy array of images, the second of labels. We need to create two Dataset objects, one for train set and one for test set (in this example, we won’t be using val set). 
~~~
train_data = tf.data.Dataset.from_tensor_slices(train)
# train_data = train_data.shuffle(10000) # if you want to shuffle your data
test_data = tf.data.Dataset.from_tensor_slices(test)
~~~
However, now we have A LOT more data. If we calculate gradient after every single data point it’d be painfully slow. Fortunately, we can process the data in batches. 
~~~
train_data = train_data.batch(batch_size)
test_data = test_data.batch(batch_size)
~~~
The next step is to get samples by iterating the two datasets. We can use the following codes to get samples.
~~~
for batch_idx, (imgs, labels) in enumerate(train_data):
   ...
~~~
## Task 1.  Using logistic regression to classify image data

You need to fill in your code to a1a.py, which is a skeleton of logistic regression using Data Loading Approach 2 as described above.

Please first read and understand a1a.py. Try to complete the code by yourself. Note that a1a.py will use functions defined in utils.py.

There is some example code logreg_example.py that uses Data Loading Approach 1.

If you meet problems and really cannot solve the problem, you can take a look at this hint list. 


## Task 2.  Improve the model of Task 1.
We got the accuracy of ~91% on our MNIST dataset with our vanilla model, which is unacceptable. The state of the art is above 99% [Links to an external site.](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html). You have full freedom to do whatever you want here, e.g. to use a different model like DNN or CNN, as long as your model is built in TensorFlow. 

You can reuse the code from part 1, but please save your code for part 2 in a separate file and name it a1b.py. 

Directly copying code from the Internet or other students will get 0 points for this assignment.

## Task 3.  Write a report
In the report, FOR EACH MODEL that you have tested in TASK1 and TASK 2, please:

describe the model;
paste a picture of the graph representation captured in TensorBoard; report the accuracy the model has achieved on MNIST dataset;
the time you spent to complete task 1 and task 2;
and interesting problems that you've met during this assignment.
Please export the report as a PDF file.

### To use TensorBoard in Linux: 
In the directory where you run the python script, type following command:
~~~
tensorboard --logdir
~~~
Then you can view the graph in following website:
~~~
localhost:6006
~~~
