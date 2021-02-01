# One-versus-all-L2-loss-SVM
Machine SVM loss without Sklearn and built in function




## Installation
Download the svm.py

## Dataset
https://www.dropbox.com/s/nf8zriqjgw02m1a/hw2-cifar-dataset.zip?dl=0.

## Import
import numpy as np

Works for all data sets


## To do 

1) SVM.make_one_versus_all_labels: 
Given an array of integer labels and the number of classes m, this function should create a 2-d
array. In this array,
each row is filled with −1, except for the entry corresponding to
the correct label, which should have entry 1. For example, if
the array of labels is [1, 0, 2] and m = 4, this function would return
the following array: [[−1, 1,−1,−1], [1,−1,−1,−1], [−1,−1, 1,−1]].
The inputs are y (a NumPy array of shape (number of labels,))
and m (an integer representing the number of classes), and the
output should be a NumPy array of shape (number of labels,m).
For this homework, m will be 10, but you should write this function
to work for any m > 2.



2) SVM.compute_loss:
Given a minibatch of examples, this function
should compute the loss function. The inputs are x (a NumPy array of shape (minibatch size, 401)), y (a NumPy array of shape
(minibatch size, 10)), and the output should be the computed
loss, a single float.


3) SVM.compute_gradient:
 Given a minibatch of examples, this
the function should compute the gradient of the loss function with
respect to the parameters w. The inputs are X (a NumPy array
of shape (minibatch size, 401)), y (a NumPy array of shape
(minibatch size, 10)), and the output should be the computed
gradient, a NumPy array of shape (401, 10), the same shape as
the parameter matrix w. (Hint: use the expressions you derived
above.
4) SVM.infer: 
Given a minibatch of examples, this function should
infer the class for each example, i.e. which class has the highest
score. The input is X (a NumPy array of shape (minibatch size, 401)
), and the output is y_inferred (a NumPy array of shape (minibatch size, 10)).
The output should be in the one-versus-all format, i.e. −1 for
each class other than the inferred class, and +1 for the inferred
class.
SVM.
5) SVM.compute_accuracy: 
Given an array of inferred labels and
an array of true labels, this function should output the accuracy
as a float between 0 and 1. The inputs are y_inferred (a
NumPy array of shape (minibatch size, 10)) and y (a NumPy array
of shape (minibatch size, 10)), and the output is a single float.



## Language used
Python

