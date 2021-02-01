import numpy as np


class SVM:
    def __init__(self,eta, C, niter, batch_size, verbose):
      self.eta = eta; self.C = C; self.niter = niter; self.batch_size = batch_size; self.verbose = verbose

    def make_one_versus_all_labels(self, y, m):
        
        a=np.zeros((len(y),m))
        if(m>2):
          for j,k in enumerate(y):
            for i in range(m):
                if(i==k):
                  a[j][i]=1
                else:
                  a[j][i]=-1
        
        return a
      
    def compute_loss(self, x, y):


       
        
        
    
        scores = x.dot(self.w)

        a = np.sum(np.maximum(0, 1 - y * scores))+.5 * self.eta * np.sum(self.w ** 2)


        loss= (a/x.shape[0])*self.C
        return loss

     
       

    def compute_gradient(self, x, y):
        """
		x : numpy array of shape (minibatch size, 401)
		y : numpy array of shape (minibatch size, 10)
		returns : numpy array of shape (401, 10)
		"""
        
        yx = (x.T).dot(y)
        yxbeta =yx*(self.w)
        compare_result= np.maximum(0, 1-yxbeta)
        part1 = (-2*self.C/len(y))*compare_result*(yx)
        part2 = 2*self.eta*self.w   
       
        return part1+part2


	# Batcher function
    def minibatch(self, iterable1, iterable2, size=1):
        l = len(iterable1)
        n = size
        for ndx in range(0, l, n):
            index2 = min(ndx + n, l)
            yield iterable1[ndx: index2], iterable2[ndx: index2]

    def infer(self, x):
        """
		x : numpy array of shape (number of examples to infer, 401)
		returns : numpy array of shape (number of examples to infer, 10)
		"""
       
        y_pred = np.dot(x, self.w)
        y_pred[y_pred > 0] = 1
        y_pred[y_pred <= 0] = -1
              
        return (y_pred)
       
        

    def compute_accuracy(self, y_inferred, y):
        """
		y_inferred : numpy array of shape (number of examples, 10)
		y : numpy array of shape (number of examples, 10)
		returns : float
		"""
        
        accuracy=np.equal(y_inferred , y)
        
        acc=np.zeros(accuracy.shape[0])
        for i,row in enumerate(accuracy):
           count=1
           for j in range(len(row)):
               if(row[j]==False):
                  count=0
               acc[i]=count
        
        return (np.sum(acc)/accuracy.shape[0])

    def fit(self, x_train, y_train, x_test, y_test):
        """
		x_train : numpy array of shape (number of training examples, 401)
		y_train : numpy array of shape (number of training examples, 10)
		x_test : numpy array of shape (number of training examples, 401)
		y_test : numpy array of shape (number of training examples, 10)
		returns : float, float, float, float
		"""
        self.num_features = x_train.shape[1]
        self.m = y_train.max() + 1
        y_train = self.make_one_versus_all_labels(y_train, self.m)
        y_test = self.make_one_versus_all_labels(y_test, self.m)
        self.w = np.zeros([self.num_features, self.m])

        for iteration in range(self.niter):
			# Train one pass through the training set
            for x, y in self.minibatch(x_train, y_train, size=self.batch_size):
                grad = self.compute_gradient(x,y)
                self.w -= self.eta * grad
		    

			# Measure loss and accuracy on training set
            train_loss = self.compute_loss(x_train,y_train)
            y_inferred = self.infer(x_train)
            train_accuracy = self.compute_accuracy(y_inferred, y_train)
			# Measure loss and accuracy on test set
            test_loss = self.compute_loss(x_test,y_test)
            y_inferred = self.infer(x_test)
            test_accuracy = self.compute_accuracy(y_inferred, y_test)
            if self.verbose:
                print("Iteration %d:" % iteration)
                print("Train accuracy: %f" % train_accuracy)
                print("Train loss: %f" % train_loss)
                print("Test accuracy: %f" % test_accuracy)
                print("Test loss: %f" % test_loss)
                print("")
        print(train_loss, train_accuracy, test_loss, test_accuracy)
        return train_loss, train_accuracy, test_loss, test_accuracy


if __name__ == "__main__":
	# Load the data files
	print("Loading data...")
	x_train = np.load("hw2-cifar-dataset/train_features.npy")
	x_test = np.load("hw2-cifar-dataset/test_features.npy")
	y_train = np.load("hw2-cifar-dataset/train_labels.npy")
	y_test = np.load("hw2-cifar-dataset/test_labels.npy")

	print("Fitting the model...")
	svm = SVM(eta=0.001, C=30, niter=200, batch_size=5000, verbose=False)
	train_loss, train_accuracy, test_loss, test_accuracy = svm.fit(x_train, y_train, x_test, y_test)

	# # to infer after training, do the following:
	y_inferred = svm.infer(x_test)

	## to compute the gradient or loss before training, do the following:
	y_train_ova = svm.make_one_versus_all_labels(y_train, 10) # one-versus-all labels
	svm.w = np.zeros([401, 10])
	grad = svm.compute_gradient(x_train, y_train_ova)
	loss = svm.compute_loss(x_train, y_train_ova)
#svm params {'eta': 0.001, 'C': 30, 'niter': 1, 'batch_size': 5000, 'verbose': False}
