from typing import List
from statistics import mode, mean
import numpy as np

from PointSet import PointSet, FeaturesTypes

label_possible_values = [False, True]

class Tree:
    """A decision Tree

    Attributes
    ----------
        points : PointSet
            The training points of the tree
        leaves : Leaf
    """


    def __init__(self,
                 features: List[List[float]],
                 labels: List[bool],
                 types: List[FeaturesTypes],
                 h: int = 1):
        """
        Parameters
        ----------
            labels : List[bool]
                The labels of the training points.
            features : List[List[float]]
                The features of the training points. Each sublist
                represents a single point. The sublists should have
                the same length as the `types` parameter, while the
                list itself should have the same length as the
                `labels` parameter.
            types : List[FeaturesTypes]
                The types of the features.
        """
        self.points = PointSet(features, labels, types)
        self.types = types
        self.h = h

        # Get best feature along which splitting provides the best Gini gain
        self.best_feature_index, _ = self.points.get_best_gain()
        self.best_feature = self.points.features[:, self.best_feature_index]
        
        best_feature_value = self.points.best_feature_value
        best_feature_possible_values = set(self.best_feature)

        # Set default decision and leaves
        self.decision = None
        self.leaves: Leaf = []
        
        # Stop condition for h=0 or ill-defined gini gains
        if (h == 0 or set(labels) == 1 or self.best_feature_index == None): 
            self.decision = mode(labels)
            return

        # Create left and right branches metrics to split the tree
        if types[self.best_feature_index] == FeaturesTypes.REAL:
            mask_best_value = (self.best_feature < self.points.get_best_threshold())
            left_values = self.best_feature[mask_best_value]
            right_values = self.best_feature[np.logical_not(mask_best_value)]
            
        else:
            mask_best_value = (self.best_feature == best_feature_value)
            left_values = [best_feature_value]
            right_values = [value for value in best_feature_possible_values if value != best_feature_value]
        
        left_features = self.points.features[mask_best_value]
        right_features = self.points.features[np.logical_not(mask_best_value)]
        
        # Stop condition if one of the branches is ill-defined
        if (len(labels) == len(left_features) or len(labels) == len(right_features)): 
            self.decision = mode(labels)
            return

        self.leaves.append(Leaf(
            feature_values=left_values,
            tree=Tree(
                features=left_features,
                labels=self.points.labels[mask_best_value],
                types=types,
                h=h-1
            )
        ))

        self.leaves.append(Leaf(
            feature_values=right_values,
            tree=Tree(
                features=right_features,
                labels=self.points.labels[np.logical_not(mask_best_value)],
                types=types,
                h=h-1
            )
        ))

    def decide(self, features: List[float]) -> bool:
        """Give the guessed label of the tree to an unlabeled point

        Parameters
        ----------
            features : List[float]
                The features of the unlabeled point.

        Returns
        -------
            bool
                The label of the unlabeled point,
                guessed by the Tree
        """
        if self.types[self.best_feature_index] == FeaturesTypes.REAL:
            # We take the decision if the node has no children
            if (self.decision != None): return self.decision
            
            # Otherwise, we call the children decision function
            elif features[self.best_feature_index] < self.points.get_best_threshold():
                if (self.leaves[0].tree.decision != None): return self.leaves[0].tree.decision
                else: return self.leaves[0].tree.decide(features)
        
            elif features[self.best_feature_index] >= self.points.get_best_threshold():
                if (self.leaves[1].tree.decision != None): return self.leaves[1].tree.decision
                else: return self.leaves[1].tree.decide(features)
        else:
            for leaf in self.leaves:
                if features[self.best_feature_index] in leaf.feature_values:
                    # We take the decision if the node has no children
                    if (leaf.tree.decision != None): return leaf.tree.decision
                    
                    # Otherwise, we call the children decision function
                    else: return leaf.tree.decide(features)

class Leaf:
    def __init__(self,
                feature_values,
                tree: Tree):

        self.feature_values = feature_values
        self.tree = tree
