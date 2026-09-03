import numpy as np
from decision_tree import Node, DecisionTree

class RandomForest():
    def __init__(self, n_trees = 10, max_depth = 12):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        
    def _bootstrap_sample(self, X, y):
        """Créer un échantillon aléatoire avec remise"""
        n_samples = X.shape[0]
        rand_indices = np.random.choice(n_samples, size = n_samples, replace = True)
        return X[rand_indices], y[rand_indices]

    def fit(self, X, y):
        """Créer et entraîner n_trees arbres"""
        for i in range(self.n_trees):
            # On crée UN échantillon Bootstrap pour cet arbre individuel
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTree(self.max_depth)
            # On entraîne cet arbre sur cet échantillon et on l'ajoute à la forêt
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        # Recolter les prédictions de chaque arbre
        tree_predictions = []
        for tree in self.trees:
            tree_predictions.append(tree.predict(X))
            
        # Grouper par ligne de donnée (n_samples, n_trees)
        votes_per_sample = np.array(tree_predictions).T
        
        # Prédiction finale (vote majoritaire)
        y_pred = []
        for votes in votes_per_sample:
            labels, counts = np.unique(votes, return_counts=True)
            majorite = labels[np.argmax(counts)]
            y_pred.append(majorite)
            
        return np.array(y_pred)
