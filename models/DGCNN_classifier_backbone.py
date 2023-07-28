# Importing necessary modules

import os
import glob
import trimesh
import warnings
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from matplotlib import pyplot as plt
import pandas as pd
import time

warnings.filterwarnings('ignore')

tf.random.set_seed(1234)


# Orthogonal Regularizer

class OrthogonalRegularizer(tf.keras.regularizers.Regularizer):
    def __init__(self, num_features, l2reg=0.001):
        self.num_features = num_features
        self.l2reg = l2reg
        self.eye = tf.eye(num_features)

    def __call__(self, x):
        x = tf.reshape(x, (-1, self.num_features, self.num_features))
        xxt = tf.tensordot(x, x, axes=(2, 2))
        xxt = tf.reshape(xxt, (-1, self.num_features, self.num_features))
        return tf.reduce_sum(self.l2reg * tf.square(xxt - self.eye))
    
    
# Custom Layers

class Kappa(tf.keras.layers.Layer):
    def __init__(self, gamma=1):
        super(Kappa, self).__init__()
        self.gamma = gamma
        
    def call(self, inputs):
        self.inputs = inputs
        
        out = get_edge_feature()(inputs)
        
        out = tf.reduce_max(out, axis=2)
        
        out = layers.Conv1D(32, kernel_size=1, padding='valid', activation='relu')(out)
    
        out = layers.BatchNormalization(momentum=0.0)(out)

        out = layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu')(out)

        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.GlobalMaxPool1D()(out)
        
        kappa_mask = layers.Dense(
            self.inputs.shape[1], activation='softmax'
        )(out)
        
        return kappa_mask

class get_edge_feature(tf.keras.layers.Layer):
    def __init__(self):
        super(get_edge_feature, self).__init__()
        
        
    def adj_matrix_mlp(self, f_pc):
        '''
        Create adj_matrix using mlp to extract critical relationships.
        Args:
            Feature_point_cloud: tensor (batch_size, num_points, num_dims)
        Returns:
        adj_matrix: (batch_size, num_points, num_points)
        '''
        
        batch_size = f_pc.shape[0]
        num_points = f_pc.shape[1]
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)
            
        x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, padding='valid',
                                  activation='relu')(f_pc)
        
        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
        
        x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='valid',
                                  activation='relu')(x)
        
        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
        
        x = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding='valid',
                                  activation='relu')(x)
        
        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)
        
        x = tf.keras.layers.Dense(num_points)(x)
        
        return x
        
    def adj_matrix(self, f_pc):
        """Create adj_matrix by computing l2 norm between features.
          Args:
            Feature_point_cloud: tensor (batch_size, num_points, num_dims)
          Returns:
        adj_matrix: (batch_size, num_points, num_points)
        """
        batch_size = f_pc.shape[0]
        f_pc = tf.squeeze(f_pc)
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)

        f_pc_T = tf.transpose(f_pc, perm=[0, 2, 1])
        f_pc_inner = tf.matmul(f_pc, f_pc_T)
        f_pc_inner = -2*f_pc_inner
        f_pc_square = tf.reduce_sum(tf.square(f_pc), axis=-1, keepdims=True)
        f_pc_square_T = tf.transpose(f_pc_square, perm=[0, 2, 1])
        
        return f_pc_square + f_pc_inner + f_pc_square_T


    def KNN(self, adj_matrix, k=20):
        """Perform KNN on feature adjacency matrix.
      Args:
        Adjacency matrix: (batch_size, num_points, num_points)
        k: int
      Returns:
        KNN: (batch_size, num_points, k)
          """
        neg_adj = -adj_matrix
        _, knn_idx = tf.nn.top_k(neg_adj, k=k)
        return knn_idx
    
    def get_edge_op(self, f_pc, knn_idx, k=20):
        """Op to compute edge feature based on underlying features.
      Args:
        Feature point cloud: (batch_size, num_points, num_dims)
        knn_idx: (batch_size, num_points, k)
        k: int
      Returns:
        Edge specific features: (batch_size, num_points, k, num_dims)
        """
        batch_size = f_pc.shape[0]
        f_pc = tf.squeeze(f_pc)
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)

        f_pc_central = f_pc

        f_pc_shape = f_pc.shape
        batch_size = f_pc_shape[0]
        num_points = f_pc_shape[1]
        num_dims = f_pc_shape[2]
            
        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
        
        
        f_pc_flat = tf.reshape(f_pc, [-1, num_dims])
        
        f_pc_neighbors = tf.gather(f_pc_flat, knn_idx+idx_)
        
        f_pc_central = tf.expand_dims(f_pc_central, axis=-2)
        
        f_pc_central = tf.tile(f_pc_central, [1, 1, k, 1])
        
        l1_distance = f_pc_neighbors - f_pc_central
        
        edge_feature = tf.concat([f_pc_central, l1_distance], axis=-1)
        
        return edge_feature

    def call(self, x):
        adj = self.adj_matrix(x)   # Shape
        
        knn_idx = self.KNN(adj)
        
        edge_feature = self.get_edge_op(x, knn_idx)
        
        return edge_feature

    
    
class get_first_edge_feature(tf.keras.layers.Layer):
    def __init__(self):
        super(get_first_edge_feature, self).__init__()
        
        
    def adj_matrix_mlp(self, f_pc):
        '''
        Create adj_matrix using mlp to extract critical relationships.
        Args:
            Feature_point_cloud: tensor (batch_size, num_points, num_dims)
        Returns:
        adj_matrix: (batch_size, num_points, num_points)
        '''

        batch_size = f_pc.shape[0]
        num_points = f_pc.shape[1]
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)

        x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, padding='valid',
                                  activation='relu')(f_pc)

        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)

        x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, padding='valid',
                                  activation='relu')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)

        x = tf.keras.layers.Conv1D(filters=512, kernel_size=1, padding='valid',
                                  activation='relu')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.0)(x)

        x = tf.keras.layers.Dense(num_points)(x)
        
        return x
   
        
    def adj_matrix(self, x):   
        """Create adj_matrix by computing l2 norm between features.
          Args:
            Feature_point_cloud: tensor (batch_size, num_points, num_dims)
          Returns:
        adj_matrix: (batch_size, num_points, num_points)
        """
        batch_size = x.shape[0]
        f_pc = x[:, :, :3]
        f_pc = tf.squeeze(f_pc)
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)

        f_pc_T = tf.transpose(f_pc, perm=[0, 2, 1])
        f_pc_inner = tf.matmul(f_pc, f_pc_T)
        f_pc_inner = -2*f_pc_inner
        f_pc_square = tf.reduce_sum(tf.square(f_pc), axis=-1, keepdims=True)
        f_pc_square_T = tf.transpose(f_pc_square, perm=[0, 2, 1])
        
        return f_pc_square + f_pc_inner + f_pc_square_T


    def KNN(self, adj_matrix, k=20):
        """Perform KNN on feature adjacency matrix.
      Args:
        Adjacency matrix: (batch_size, num_points, num_points)
        k: int
      Returns:
        KNN: (batch_size, num_points, k)
          """
        neg_adj = -adj_matrix
        _, knn_idx = tf.nn.top_k(neg_adj, k=k)
        return knn_idx
    
    def get_edge_op(self, x, knn_idx, k=20):
        """Op to compute edge feature based on underlying features.
      Args:
        Feature point cloud: (batch_size, num_points, num_dims)
        knn_idx: (batch_size, num_points, k)
        k: int
      Returns:
        Edge specific features: (batch_size, num_points, k, num_dims)
        """
        batch_size = x.shape[0]

        f_pc = x
        
        f_pc = tf.squeeze(f_pc)
        if batch_size == 1:
            f_pc = tf.expand_dims(f_pc, 0)

        f_pc_central = f_pc

        f_pc_shape = f_pc.shape

        batch_size = f_pc_shape[0]
        num_points = f_pc_shape[1]
        num_dims = f_pc_shape[2]
            
        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 
        
        
        f_pc_flat = tf.reshape(f_pc, [-1, num_dims])
        
        f_pc_neighbors = tf.gather(f_pc_flat, knn_idx+idx_)
        
        f_pc_central = tf.expand_dims(f_pc_central, axis=-2)
        
        f_pc_central = tf.tile(f_pc_central, [1, 1, k, 1])
        
        l1_distance = f_pc_neighbors - f_pc_central
        
        l2_distance = tf.math.square(l1_distance)
        
        l3_distance = tf.math.pow(l1_distance, 3)

        edge_feature = tf.concat([f_pc_central, l1_distance], 
                                 axis=-1)
        
        return edge_feature

    def call(self, x):
        adj = self.adj_matrix_mlp(x)   # Shape
        
        knn_idx = self.KNN(adj)
        
        edge_feature = self.get_edge_op(x, knn_idx)
        
        return edge_feature

    
    
class KNN_Downsampler(tf.keras.layers.Layer):
    '''
    Uses MLP to approximate a Downsampler by convolving over edge features.
    The Downsampler calculates an importance weight mask thus assigning contribution score 
    towards final prediction for each point.
    '''
    
    def __init__(self, num_points, reduce_num=1024):
        super(KNN_Downsampler, self).__init__()
        self.num_points = num_points
        self.reduce_num = reduce_num
        self.bias = keras.initializers.Constant(np.eye(num_points).flatten())
        self.reg = OrthogonalRegularizer(num_points)
        
        
    def ds_op(self, weight_mask):
        idx = tf.argsort(weight_mask, axis=1, direction='DESCENDING')

        multidim = tf.gather(self.inputs, idx, axis=1)

        ds_points = tf.stack(
            [multidim[i, i, :self.reduce_num, :] for i in tf.range(self.inputs.shape[0])], axis=0
        )

        return ds_points
     
    
    def call(self, inputs):
        self.inputs = inputs
        
        out = get_edge_feature()(inputs)
        
        out = tf.reduce_max(out, axis=2)
        
        out = layers.Conv1D(32, kernel_size=1, padding='valid', activation='relu')(out)
    
        out = layers.BatchNormalization(momentum=0.0)(out)

        out = layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu')(out)

        out = layers.BatchNormalization(momentum=0.0)(out)

        out = layers.Conv1D(512, kernel_size=1, padding='valid', activation='relu')(out)

        out = layers.BatchNormalization(momentum=0.0)(out)

        out = layers.GlobalMaxPool1D()(out)
        
        weight_mask = layers.Dense(
            self.num_points, activation='softmax'
        )(out)
        
        ds_points = self.ds_op(weight_mask)
        
        return ds_points
    
    
        
    
class affine_net(tf.keras.layers.Layer):
    '''
    An mlp based transformer for converting input point cloud to similar canonical shape.
    '''
    def __init__(self, num_features):
        self.num_features = num_features
        self.bias = keras.initializers.Constant(np.eye(num_features).flatten())
        self.reg = OrthogonalRegularizer(num_features)
        super(affine_net, self).__init__()

    
    def call(self, inputs):
        out = layers.Conv1D(32, kernel_size=1, padding='valid', activation='relu')(inputs)
        
        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.Conv1D(64, kernel_size=1, padding='valid', activation='relu')(out)
    
        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.Conv1D(512, kernel_size=1, padding='valid', activation='relu')(out)
        
        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.GlobalMaxPooling1D()(out)
        
        out = layers.Dense(256)(out)
        
        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.Dense(128)(out)
        
        out = layers.BatchNormalization(momentum=0.0)(out)
        
        out = layers.Dense(
            self.num_features*self.num_features, kernel_initializer='zeros', bias_initializer=self.bias, activity_regularizer=self.reg,
        )(out)
        
        affine_feat = layers.Reshape((self.num_features, self.num_features))(out)
        
        return layers.Dot(axes=(2,1))([inputs, affine_feat])        
        

'''
DMS-DGCNN : Acronynm for Differential Multi-Spectral DGCNN
'''

class DMS_DGCNN(tf.keras.Model):
    def __init__(self, NUM_CHANNELS):
        super(DMS_DGCNN, self).__init__()
        self.transformer = affine_net(num_features=3)
        self.downsampler = KNN_Downsampler(num_points=300, reduce_num=300)
        self.conv1 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')
        self.bn1 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.conv2 = tf.keras.layers.Conv1D(filters=64, kernel_size=1, activation='relu')
        self.bn2 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.conv3 = tf.keras.layers.Conv1D(filters=128, kernel_size=1, activation='relu')
        self.bn3 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.conv4 = tf.keras.layers.Conv1D(filters=256, kernel_size=1, activation='relu')
        self.bn4 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.conv5 = tf.keras.layers.Conv1D(filters=256, kernel_size=1, activation='relu')
        self.pool1 = tf.keras.layers.GlobalMaxPooling1D()
        self.pool2 = tf.keras.layers.GlobalAveragePooling1D()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.bn5 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.bn6 = tf.keras.layers.BatchNormalization(momentum=0.0)
        self.dense3 = tf.keras.layers.Dense(NUM_CHANNELS, activation='softmax')
#         self.getIntensityFeature = get_intensity_edge_feature()
        self.getFeature = get_edge_feature()
        
        
    def call(self, inputs):
        ds_inputs = self.downsampler(inputs)
        
#         edge_intensity_feature = self.getIntensityFeature(ds_inputs)
        
#         out = self.conv1(edge_intensity_feature)

#         out = self.bn1(out)

#         feature_I = tf.reduce_max(out, axis=2)

        edge_feature = self.getFeature(ds_inputs)
        
        out = self.conv1(edge_feature)

        out = self.bn1(out)

        feature1 = tf.reduce_max(out, axis=2)

        edge_feature = self.getFeature(feature1)
    
        out = self.conv2(edge_feature)
        
        out = self.bn2(out)
        
        feature2 = tf.reduce_max(out, axis=2)

        edge_feature = self.getFeature(feature2)

        out = self.conv3(edge_feature)

        out = self.bn3(out)

        feature3 = tf.reduce_max(out, axis=2)
        
#         print(feature3.shape)

#         edge_feature = self.getFeature(feature3)

#         out = self.conv4(edge_feature)

#         out = self.bn4(out)

#         feature4 = tf.reduce_max(out, axis=2)
        
#         features = tf.concat((feature1, feature2, feature3, feature4), axis=2)

        out = self.conv5(feature3)
        
        feature5 = self.pool1(out)
        
#         feature6 = self.pool2(out)

#         features = tf.concat((feature5, feature6), 1)

        out = self.dense1(feature5)

        out = self.bn5(out)

        out = self.dense2(out)

        out = self.bn6(out)

        out = self.dense3(out)

        outputs = out

        return outputs