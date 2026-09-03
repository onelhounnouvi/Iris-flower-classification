import numpy as np

class Node():
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
  
    
class DecisionTree():
    def __init__(self, max_depth = 12):
        self.root = None
        self.max_depth = max_depth
        
    def calculate_gini(self, y):
        """Retourne le score d'impureté de Gini"""
        if len(y) == 0:
            return 0
        else:
            _, counts = np.unique(y, return_counts=True)
            N = len(y)
            return 1 - np.sum((counts/N)**2)
        
    def information_gain(self, y, y_left, y_right):
        """Retourne le gain d'information dans les noeuds fils par rapport au noeud parent"""
        W_left = len(y_left)/len(y)
        W_right = len(y_right)/len(y)
        return self.calculate_gini(y) - W_left*self.calculate_gini(y_left) - W_right*self.calculate_gini(y_right)
  
        
    def _split(self, X_column, threshold):
        """Renvoie les indices des données qui vont à gauche ou à droite de l'arbre, par rapport au seuil"""
        left = np.where(X_column <= threshold)[0]
        right = np.where(X_column > threshold)[0]
        return left, right
    
    def get_best_split(self, X, y):
        """Renvoie la meilleure division du noeud parent en comparant les gains d'information"""
        best_gain = -1
        best_feature_idx = None
        best_threshold = None
        
        for feature_idx in range(X.shape[1]):
            X_column = X[:,feature_idx]
            candidats = np.unique(X_column)
            for threshold in candidats:
                left, right = self._split(X_column, threshold)
                y_left = y[left]
                y_right = y[right]
                gain = self.information_gain(y, y_left, y_right)
                if gain > best_gain:
                    best_gain = gain
                    best_feature_idx = feature_idx
                    best_threshold = threshold
                    
        return best_gain, best_feature_idx, best_threshold
    
    def _build_tree(self, X, y, current_depth=0):
        """Construit récursivement l'arbre de décision"""
        if len(set(y)) <= 1 or current_depth == self.max_depth:
            labels, counts = np.unique(y, return_counts=True)
            node = Node()
            node.value = labels[np.argmax(counts)]
            return node
        
        gain, feature_idx, threshold = self.get_best_split(X, y)
        
        if gain <= 0:
            labels, counts = np.unique(y, return_counts=True)
            node = Node()
            node.value = labels[np.argmax(counts)]
            return node
        
        X_column = X[:, feature_idx]
        left, right = self._split(X_column, threshold)
        X_left, y_left = X[left], y[left]
        X_right, y_right = X[right], y[right]
        
        node = Node()
        node.feature = feature_idx
        node.threshold = threshold
        node.left = self._build_tree(X_left, y_left, current_depth + 1)
        node.right = self._build_tree(X_right, y_right, current_depth + 1)
        
        return node
        
    def fit(self, X, y):
        self.root = self._build_tree(X, y)
        
    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
