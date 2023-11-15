from typing import List, Tuple

from enum import Enum
import numpy as np

class FeaturesTypes(Enum):
    """Enumerate possible features types"""
    BOOLEAN=0
    CLASSES=1
    REAL=2

class PointSet:
    """A class representing set of training points.

    Attributes
    ----------
        types : List[FeaturesTypes]
            Each element of this list is the type of one of the
            features of each point
        features : np.array[float]
            2D array containing the features of the points. Each line
            corresponds to a point, each column to a feature.
        labels : np.array[bool]
            1D array containing the labels of the points.
    """
    def __init__(self, features: List[List[float]], labels: List[bool], types: List[FeaturesTypes]):
        """
        Parameters
        ----------
        features : List[List[float]]
            The features of the points. Each sublist contained in the
            list represents a point, each of its elements is a feature
            of the point. All the sublists should have the same size as
            the `types` parameter, and the list itself should have the
            same size as the `labels` parameter.
        labels : List[bool]
            The labels of the points.
        types : List[FeaturesTypes]
            The types of the features of the points.
        """
        self.types = types
        self.features = np.array(features)
        self.labels = np.array(labels)
    
    def get_gini(self) -> float:
        """Computes the Gini score of the set of points

        Returns
        -------
        float
            The Gini score of the set of points
        """
        true_labels_count = 0
        num_of_labels = len(self.labels)
        
        if not num_of_labels: return 1

        for label in self.labels:
            true_labels_count += 1 if label == True else 0       
        
        prob_of_true = true_labels_count / num_of_labels
        prob_of_false = 1 - prob_of_true
        
        gini = 1 - (prob_of_true**2 + prob_of_false**2)
        
        return gini

    def get_best_gain(self) -> Tuple[int, float]:
        """Compute the feature along which splitting provides the best gain

        Returns
        -------
        int
            The ID of the feature along which splitting the set provides the
            best Gini gain.
        float
            The best Gini gain achievable by splitting this set along one of
            its features.
        """
        num_of_features = len(self.features[0])
        splits = []
        
        for feature_index in range(num_of_features):
            feature = self.features[:, feature_index]
            feature_possible_values = set(feature)
            
            for value in feature_possible_values:
                mask_value = (feature < value) if self.types[feature_index] == FeaturesTypes.REAL else (feature == value)
                
                point_set_left = PointSet(
                    features = self.features[mask_value],
                    labels = self.labels[mask_value],
                    types = self.types
                )
                
                point_set_right = PointSet(
                    features = self.features[np.logical_not(mask_value)],
                    labels = self.labels[np.logical_not(mask_value)],
                    types = self.types
                )

                gini_split = (len(point_set_left.labels) * point_set_left.get_gini() + len(point_set_right.labels) * point_set_right.get_gini()) / len(self.labels)             
                gini_gain = self.get_gini() - gini_split
                splits.append([feature_index, value, gini_gain])
        
        all_gini_gains = [row[2] for row in splits]

        if np.all(all_gini_gains == 0):
            return (None, None)

        best_split_index = np.argmax(all_gini_gains)
        
        best_feature_index = splits[best_split_index][0]
        best_feature_value = splits[best_split_index][1]
        best_gini_gain = splits[best_split_index][2]
        
        self.best_feature_index = best_feature_index
        self.best_feature_value = best_feature_value
        
        return (best_feature_index, best_gini_gain)
            
    def get_best_threshold(self) -> float:
        if (self.types[self.best_feature_index] == FeaturesTypes.REAL):
            mask_value = self.features[:, self.best_feature_index] < self.best_feature_value
                
            features_left =  self.features[mask_value]
            features_right = self.features[np.logical_not(mask_value)]
            
            max_left = max(features_left[:, self.best_feature_index])
            min_right = min(features_right[:, self.best_feature_index])

            return (max_left + min_right) / 2
        elif (self.types[self.best_feature_index] == FeaturesTypes.CLASSES):
            return self.best_feature_value
        else:
            return None
