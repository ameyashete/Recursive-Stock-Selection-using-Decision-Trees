# In this file, we will build the builing blocks for decision trees. 
import numpy as np
from collections import Counter
import math


# Define a class for node of a tree
class Node:
	def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
		# Which feature this node was divided with
		self.feature = feature
		# At what threshold was this node divided at
		self.threshold = threshold
		self.left = left
		self.right = right
		self.value = value

	def is_leaf_node(self):
		"""
		This function will help us determine if a node is the leaf node. 
		
		Only leaf nodes will have values. This is because the leaf nodes will be 
		the prediction we make for the model. 
		"""
		return self.value is not None


# Define a class for the decision tree
class DecisionTree:
	def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
		
		self.min_samples_split=min_samples_split
		
		# Max depth will be how deep you want to take the tree. 
		self.max_depth=max_depth
		
		# n_features is an input which will determine the length of the subset of features you want to use. 
		
		# This will ensure that we have some randomness when building our tree. This is most commonly 
		# used with random forests and other ensemble methods. This is to ensure that the many trees 
		# we are building are decorrelated from each other, reducing overfitting and avoiding multicollinearity.
		self.n_features=n_features

		self.root=None


	def fit(self, X, y):

		self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
		# Once the entire tree has been grown, we will return the root node since it will contain all of the information
		# of the rest of the tree. 
		self.root = self._grow_tree(X, y)


	def _grow_tree(self, X, y, depth=0):
		n_sample, n_features = X.shape
		n_labels = np.unique

		# check stopping criteria
		# When we have 1 label, it is a child ndoe; more depth than allowed; fewer samples than required to split, we stop.
		if (depth > self.max_depth or n_sample < self.min_samples_split or n_labels = 1):
			leaf_value = self._most_common_label(y)
			return Node(value=leaf_value)

		# This will return an array of indices. So, from the total number of features we have, we choose n_features defined by the 
		# decision tree class. We will send this feat_idx to our best split function.
		feat_idx = np.random.choice(n_features, self.n_features, replace=False)


		# check best split
		best_thresh, best_feature = self._best_split(X, y, feat_idx)

		# create child nodes
		left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
		left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
		right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
		return Node(best_feature, best_thresh, left, right)

	def _most_common_label(self, y):
		count = Counter(y)
		most_common = count.most_common(1)[0][0]
		return most_common

	def _best_split(self, X, y, feat_idx):
		"""
		In addition to the X and y datasets, this function will also take in feature_idx which is feature index. 
		The goal of feat_indx is so that we an incorporate some randomness into the features we select. We will
		select a subset of the overall features we have. 

		We will find the best split using the information gain and entropy. 
		"""
		best_gain = -1
		split_idx, split_threshold = None, None

		for idx in feat_idx:
			X_column = X[:, idx]
			# This thresholds will be for continuous values as well. If we have 100 unique, continuous values, we will have 
			# 100 thresholds. 
			thresholds = np.unique(X_column)
			for thr in thresholds: 
				gain = self.information_gain(y, X_column, thr)

				if gain > best_gain:
					best_gain = gain
					split_idx = idx
					split_threshold = thr

		return split_threshold, split_idx


	def information_gain(self, y, X_column, thr):
		"""
		- This function will help us in calculating the information gain for the data
		- IG = Entropy(parent) - [Weight Average] * Entropy(children)
		- IG values range from 0 to 0.5. A higher value signifies higher IG.  
		"""

		# calculate the parent entropy
		parent_entropy = self.entrop(y)

		# calculate the chld entropy
		left_idxs, right_idxs = self._split(X_column, thr)

		if len(left_idx) == 0 or len(right_idx) == 0:
			return 0

		# calculate the weighted average 
		n = len(y)
		n_l, n_r = len(left_idxs), len(right_idxs)
		e_l, e_r = self.entropy(y[left_idxs]), self_entropy(y[right_idxs])
		child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

		# calculate IG  
		ig = parent_entropy - child_entropy
		return ig


	def _split(self, X_column, split_thresh):
		
		left_idxs = np.argwhere(X_column < split_thresh).flatten()
		right_idxs = np.argwhere(X_column >= split_thresh).flatten()

		return left_idxs, right_idxs

	def entropy(self, y):
		"""
		- This function will calculate the entropy of our data. 
		- Entropy = Summation{i=1 to c} - p_i * log_2(p_i)
		- In the above formula, p_i is essentially the probability of the data points belonging to that class. 
		- Thus, if we have continuous data, we will classify the points to being less than or greater the threshold 
		  and we will calculate the entropy accordingly. 
		"""
		prob_labels = Counter(y)
		final_entropy = 0
		for k, v in count_labels:
			prob_label = v/len(y)
			entropy += -prob_label * math.log2(prob_label)

		return final_entropy


	# We note that the training process for this tree is when we actually create the tree while the prediction 
	# process is when we input the x values into the tree which will give us the predicted class. This one value 
	# will traverse the entire tree and the decisions for it will be made accordingly until it reaches a leaf node
	# which will classify it. 

	def predict(self, X):
		return np.array([self._tree_traversal(x) for x in X])


	def _tree_traversal(self, x, node):
		if node.is_leaf_node():
			return node.value

		if x[node.feature] <= node.threshold:
			return self._tree_traversal(x, node.left)
		else:
			return self._tree_traversal(x, node.right)































