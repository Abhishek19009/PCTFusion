from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import time
import math


def gather(x, idx, method=2):
    """
    implementation of a custom gather operation for faster backwards.
    :param x: input with shape [N, D_1, ... D_d]
    :param idx: indexing with shape [n_1, ..., n_m]
    :param method: Choice of the method
    :return: x[idx] with shape [n_1, ..., n_m, D_1, ... D_d]
    """

    if method == 0:
        return x[idx]
    elif method == 1:
        x = x.unsqueeze(1)
        x = x.expand((-1, idx.shape[-1], -1))
        idx = idx.unsqueeze(2)
        idx = idx.expand((-1, -1, x.shape[-1]))
        return x.gather(0, idx)
    elif method == 2:
        for i, ni in enumerate(idx.size()[1:]):
            x = x.unsqueeze(i+1)
            new_s = list(x.size())
            new_s[i+1] = ni
            x = x.expand(new_s)
        n = len(idx.size())
        for i, di in enumerate(x.size()[n:]):
            idx = idx.unsqueeze(i+n)
            new_s = list(idx.size())
            new_s[i+n] = di
            idx = idx.expand(new_s)
        return x.gather(0, idx)
    

    else:
        raise ValueError('Unkown method')
        
        
def closest_pool(x, upsamples):
    """
    Pooling from closest neighbors.
    Only first column is used for pooling.
    """
    
    # Add last row with zero features
    
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)
    
    # Get features
    return gather(x, upsamples[:, 0])



def max_pool(x, inds):
    """
    Pools features with the maximum values.
    :param x: [n1, d] features matrix
    :param inds: [n2, max_num] pooling indices
    :return: [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

    # Get all features for each pooling location [n2, max_num, d]
    pool_features = gather(x, inds)

    # Pool the maximum [n2, d]
    max_features, _ = torch.max(pool_features, 1)
    return max_features


def block_decider(block_name,
                  radius,
                  in_dim,
                  out_dim,
                  layer_ind,
                  config):

    if block_name == 'unary':
        return LinearBlock(in_dim, out_dim, config.use_batch_norm, config.batch_norm_momentum)

    elif block_name in ['simple',
                        'simple_deformable',
                        'simple_invariant',
                        'simple_equivariant',
                        'simple_strided',
                        'simple_deformable_strided',
                        'simple_invariant_strided',
                        'simple_equivariant_strided']:
        return SimpleBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name in ['resnetb',
                        'resnetb_final',
                        'resnetb_invariant',
                        'resnetb_equivariant',
                        'resnetb_deformable',
                        'resnetb_strided',
                        'resnetb_deformable_strided',
                        'resnetb_equivariant_strided',
                        'resnetb_invariant_strided']:
        return ResNetBlock(block_name, in_dim, out_dim, radius, layer_ind, config)

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return MaxPoolBlock(layer_ind)

    elif block_name == 'global_average':
        return GlobalAverageBlock()

    elif block_name == 'nearest_upsample':
        return NearestUpsampleBlock(layer_ind)

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)




class ConvOP(nn.Module):
    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                deformable=False, modulated=False):
        """
        Initialize parameters for ConvOP.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        
        super(ConvOP, self).__init__()
        
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        
        
        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        
        # Reset Parameters
        self.reset_parameters()
        
        # Initialize kernel points
        self.kernel_points = self.init_KP()
        
        return
    
    
    
    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return
    
    
    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        
        # Create one kernel disposition (as numpy array). 
        K_points_numpy = load_kernels(self.radius,
                                     self.K,
                                     dimension=self.p_dim,
                                     fixed=self.fixed_kernel_points)
        
        
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                        requires_grad=False)
    
    
    def forward(self, q_pts, s_pts, neighb_inds, x):
        if self.deformable:
            pass

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
#         neighbors = s_pts[neighb_inds, :]
        neighbors = gather(s_pts, neighb_inds)

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:
            pass
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            pass

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        
        kernel_outputs = torch.matmul(weighted_features, self.weights)

        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0)
    
    
class PosConvOP(nn.Module):
    def __init__(self, kernel_size, p_dim, in_channels, out_channels, KP_extent, radius,
                fixed_kernel_points='center', KP_influence='linear', aggregation_mode='sum',
                deformable=False, modulated=False):
        """
        Initialize parameters for ConvOP.
        :param kernel_size: Number of kernel points.
        :param p_dim: dimension of the point space.
        :param in_channels: dimension of input features.
        :param out_channels: dimension of output features.
        :param KP_extent: influence radius of each kernel point.
        :param radius: radius used for kernel point init. Even for deformable, use the config.conv_radius
        :param fixed_kernel_points: fix position of certain kernel points ('none', 'center' or 'verticals').
        :param KP_influence: influence function of the kernel points ('constant', 'linear', 'gaussian').
        :param aggregation_mode: choose to sum influences, or only keep the closest ('closest', 'sum').
        :param deformable: choose deformable or not
        :param modulated: choose if kernel weights are modulated in addition to deformed
        """
        
        super(PosConvOP, self).__init__()
        
        # Save parameters
        self.K = kernel_size
        self.p_dim = p_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.KP_extent = KP_extent
        self.fixed_kernel_points = fixed_kernel_points
        self.KP_influence = KP_influence
        self.aggregation_mode = aggregation_mode
        self.deformable = deformable
        self.modulated = modulated
        
        
        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K, in_channels, out_channels), dtype=torch.float32), requires_grad=True)
        
        self.pos_mlp = nn.Linear(4, out_channels//2)
#         self.pos_mlp_bn = nn.BatchNorm1d(out_dim//2, momentum=self.bn_momentum)
        
        self.feat_mlp = nn.Linear(in_channels, out_channels // 2)
#         self.feat_mlp_bn = nn.BatchNorm1d(out_dim//2, momentum=self.bn_momentum)
        
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(out_channels)
        
        
        
        # Reset Parameters
        self.reset_parameters()
        
        # Initialize kernel points
        self.kernel_points = self.init_KP()
        
        return
    
    
    
    def reset_parameters(self):
        kaiming_uniform_(self.weights, a=math.sqrt(5))
        return
    
    
    def init_KP(self):
        """
        Initialize the kernel point positions in a sphere
        :return: the tensor of kernel points
        """
        
        # Create one kernel disposition (as numpy array). 
        K_points_numpy = load_kernels(self.radius,
                                     self.K,
                                     dimension=self.p_dim,
                                     fixed=self.fixed_kernel_points)
        
        
        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                        requires_grad=False)
    
    def pos_embedding(self, neighbors):
        relative_xyz = neighbors
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz], dim=-1)
        
        return relative_feature
    
    
    def forward(self, q_pts, s_pts, neighb_inds, x):
        if self.deformable:
            pass

        else:
            offsets = None
            modulations = None

        ######################
        # Deformed convolution
        ######################

        # Add a fake point in the last row for shadow neighbors
        s_pts = torch.cat((s_pts, torch.zeros_like(s_pts[:1, :]) + 1e6), 0)

        # Get neighbor points [n_points, n_neighbors, dim]
#         neighbors = s_pts[neighb_inds, :]
        neighbors = gather(s_pts, neighb_inds)

        # Center every neighborhood
        neighbors = neighbors - q_pts.unsqueeze(1)

        # Apply offsets to kernel points [n_points, n_kpoints, dim]
        if self.deformable:
            self.deformed_KP = offsets + self.kernel_points
            deformed_K_points = self.deformed_KP.unsqueeze(1)
        else:
            deformed_K_points = self.kernel_points

        # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
        neighbors.unsqueeze_(2)
        differences = neighbors - deformed_K_points

        # Get the square distances [n_points, n_neighbors, n_kpoints]
        sq_distances = torch.sum(differences ** 2, dim=3)

        # Optimization by ignoring points outside a deformed KP range
        if self.deformable:
            pass
        else:
            new_neighb_inds = neighb_inds

        # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
        if self.KP_influence == 'constant':
            # Every point get an influence of 1.
            all_weights = torch.ones_like(sq_distances)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'linear':
            # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
            all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
            all_weights = torch.transpose(all_weights, 1, 2)

        elif self.KP_influence == 'gaussian':
            # Influence in gaussian of the distance.
            sigma = self.KP_extent * 0.3
            all_weights = radius_gaussian(sq_distances, sigma)
            all_weights = torch.transpose(all_weights, 1, 2)
        else:
            raise ValueError('Unknown influence function type (config.KP_influence)')

        # In case of closest mode, only the closest KP can influence each point
        if self.aggregation_mode == 'closest':
            neighbors_1nn = torch.argmin(sq_distances, dim=2)
            all_weights *= torch.transpose(nn.functional.one_hot(neighbors_1nn, self.K), 1, 2)

        elif self.aggregation_mode != 'sum':
            raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

        # Add a zero feature for shadow neighbors
        x = torch.cat((x, torch.zeros_like(x[:1, :])), 0)

        # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
        neighb_x = gather(x, new_neighb_inds)

        # Apply distance weights [n_points, n_kpoints, in_fdim]
        weighted_features = torch.matmul(all_weights, neighb_x)

        # Apply modulations
        if self.deformable and self.modulated:
            pass

        # Apply network weights [n_kpoints, n_points, out_fdim]
        weighted_features = weighted_features.permute((1, 0, 2))
        
        kernel_outputs = torch.matmul(weighted_features, self.weights)
        
        pos_embed = self.pos_embedding(neighbors.squeeze(2))
        pos_embed = self.pos_mlp(pos_embed)
        
        
        feature_embed = self.feat_mlp(neighb_x)
        fused_embed = torch.cat([feature_embed, pos_embed], dim=-1)
        fused_embed = self.adaptive_avg_pool(fused_embed)
        
        # Convolution sum [n_points, out_fdim]
        return torch.sum(kernel_outputs, dim=0) + torch.sum(fused_embed, dim=1)

    
class PosEncoding(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum):
        super(PosEncoding, self).__init__()
        self.use_bn = use_bn
        self.bn_momentum = bn_momentum
#         self.pos_mlp = LinearBlock(10, out_dim//2, self.use_bn, self.bn_momentum)
        self.pos_mlp = nn.Linear(10, out_dim//2)
        self.pos_mlp_bn = nn.BatchNorm1d(out_dim//2, momentum=self.bn_momentum)
        self.pos_mlp_relu = nn.ReLU(inplace=True)
        
#         self.feat_mlp = nn.Linear(out_dim//2, out_dim//2)
        self.feat_mlp = nn.Linear(in_dim, out_dim // 2)
        self.feat_mlp_bn = nn.BatchNorm1d(out_dim//2, momentum=self.bn_momentum)
        self.feat_mlp_relu = nn.ReLU(inplace=True)
        
        self.adaptive_avg_pool = nn.AdaptiveAvgPool1d(out_dim)
        
    
    def forward(self, points, neighbors, feats):
        pos_embed = self.pos_embedding(points, neighbors)
        pos_embed = self.pos_mlp(pos_embed)
        pos_embed = self.pos_mlp_bn(pos_embed.transpose(1,2)).transpose(1,2)
        pos_embed = self.pos_mlp_relu(pos_embed)
        feature_embed = self.feature_embedding(feats, neighbors)
        feature_embed = self.feat_mlp(feature_embed)
        feature_embed = self.feat_mlp_bn(feature_embed.transpose(1,2)).transpose(1,2)
        feature_embed = self.feat_mlp_relu(feature_embed)
        fused_embed = torch.cat([feature_embed, pos_embed], dim=-1)
        fused_embed = self.adaptive_avg_pool(fused_embed)
        return torch.sum(fused_embed, dim=1)
    
    
    def feature_embedding(self, feature, neighbors):
        unpadded_feature = feature
        padded_features = torch.cat((feature, torch.zeros_like(feature[:1, :])), 0)
        neighb_features = gather(padded_features, neighbors)
        return neighb_features
    
    
    def pos_embedding(self, points, neighbors):
        unpadded_points = points
        padded_points = torch.cat((points, torch.zeros_like(points[:1, :]) + 1e6), 0)
        neighb_xyz = gather(padded_points, neighbors)

        xyz_tile = unpadded_points.unsqueeze(1).repeat(1,  neighbors.shape[-1], 1)

        relative_xyz = xyz_tile - neighb_xyz
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=-1, keepdim=True))
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighb_xyz], dim=-1)
        
        return relative_feature
    
    
class AttentionHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum):
        super(AttentionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        self.q_proj = nn.Linear(in_dim, out_dim)
        self.k_proj = nn.Linear(in_dim, out_dim)
        self.v_proj = nn.Linear(in_dim, out_dim)
#         self.q_proj = nn.Conv1d(in_dim, out_dim, kernel_size=(1,))
#         self.k_proj = nn.Conv1d(in_dim, out_dim, kernel_size=(1,))
#         self.v_proj = nn.Conv1d(in_dim, out_dim, kernel_size=(1,))
#         self.p_proj = LinearBlock_V2(3, out_dim, use_bn, bn_momentum, no_relu=True)
        
#         self.w_proj = LinearBlock_V2(out_dim, out_dim, use_bn, bn_momentum, no_relu=True)
        
        self.p_proj = nn.Sequential(nn.Linear(3, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))
        
        self.w_proj = nn.Sequential(nn.Linear(out_dim, out_dim), nn.BatchNorm1d(out_dim), nn.ReLU(inplace=True))
        
        self.softmax = nn.Softmax(dim=1)
        
    
    def forward(self, points, neighbors, feats):
#         q_feats = self.q_proj(feats.transpose(1,2)).transpose(1,2).squeeze(0)
#         k_feats = self.k_proj(feats.transpose(1,2)).transpose(1,2).squeeze(0)
#         v_feats = self.v_proj(feats.transpose(1,2)).transpose(1,2).squeeze(0)
        
        q_feats = self.q_proj(feats)
        k_feats = self.k_proj(feats)
        v_feats = self.v_proj(feats)
        
        points = torch.cat((points, torch.zeros_like(points[:1, :]) + 1e6), 0)
        k_feats = torch.cat((k_feats, torch.zeros_like(k_feats[:1, :])), 0)
        v_feats = torch.cat((v_feats, torch.zeros_like(v_feats[:1, :])), 0)
        
        
        p_feats = gather(points, neighbors)
        k_feats = gather(k_feats, neighbors)
        v_feats = gather(v_feats, neighbors)
        

        
        for i, layer in enumerate(self.p_proj):
            if i == 1:
                p_feats = layer(p_feats.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            else:
                p_feats = layer(p_feats)
        
        
#         w_feats = k_feats - q_feats.unsqueeze(1) + p_feats
        w_feats = (k_feats * q_feats.unsqueeze(1)) + p_feats
        
        for i, layer in enumerate(self.w_proj):
            if i == 1:
                w_feats = layer(w_feats.transpose(1,2).contiguous()).transpose(1,2).contiguous()
            else:
                w_feats = layer(w_feats)
                
        
        w_feats = self.softmax(w_feats)
        
        att_feats = ((v_feats + p_feats) * w_feats).sum(1)
        
        return att_feats
    
    
class GlobalAttentionHead(nn.Module):
    """
    Self attention performed globally.
    """
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum):
        super(GlobalAttentionHead, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.use_bn = use_bn
        self.bn_momentum = bn_momentum
        self.q_mlp = nn.Linear(in_dim, out_dim//4)
        self.k_mlp = nn.Linear(in_dim, out_dim//4)
        self.v_mlp = nn.Linear(in_dim, out_dim)
        
        self.residual = nn.Linear(in_dim, out_dim)
        
        self.batch_norm = BatchNormBlock(out_dim, use_bn, bn_momentum)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
        
    def forward(self, x):
        q_feats = self.q_mlp(x) #(N, out_dim//4)
        k_feats = self.k_mlp(x).transpose(0,1)   # (out_dim//4, N)
        v_feats = self.v_mlp(x)  # (N, out_dim)
        
        # Compute attention
        energy = torch.matmul(q_feats, k_feats) / math.sqrt(self.out_dim // 4)
        attention = self.softmax(energy)
        
        at_feats = torch.matmul(energy, v_feats)
        at_feats = self.activation(self.batch_norm(at_feats))
        
        residual = self.residual(x)
        
        return residual + at_feats
        
        
        
    
    
class BatchNormBlock(nn.Module):

    def __init__(self, in_dim, use_bn, bn_momentum):
        """
        Initialize a batch normalization block. If network does not use batch normalization, replace with biases.
        :param in_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """
        super(BatchNormBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.in_dim = in_dim
        if self.use_bn:
            self.batch_norm = nn.BatchNorm1d(in_dim, momentum=bn_momentum)
            #self.batch_norm = nn.InstanceNorm1d(in_dim, momentum=bn_momentum)
        else:
            self.bias = Parameter(torch.zeros(in_dim, dtype=torch.float32), requires_grad=True)
        return

    def reset_parameters(self):
        nn.init.zeros_(self.bias)

    def forward(self, x):
        if self.use_bn:

            x = x.unsqueeze(2)
            x = x.transpose(0, 2)
            x = self.batch_norm(x)
            x = x.transpose(0, 2)
            return x.squeeze()
        else:
            return x + self.bias


    
class LinearBlock(nn.Module):

    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        """
        Initialize a standard unary block with its ReLU and BatchNorm.
        :param in_dim: dimension input features
        :param out_dim: dimension input features
        :param use_bn: boolean indicating if we use Batch Norm
        :param bn_momentum: Batch norm momentum
        """

        super(LinearBlock, self).__init__()
        self.bn_momentum = bn_momentum
        self.use_bn = use_bn
        self.no_relu = no_relu
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.batch_norm = BatchNormBlock(out_dim, self.use_bn, self.bn_momentum)
        if not no_relu:
            self.leaky_relu = nn.LeakyReLU(0.1)
        return

    def forward(self, x, batch=None):
        x = self.mlp(x)
        x = self.batch_norm(x)
        if not self.no_relu:
            x = self.leaky_relu(x)
        return x

    def __repr__(self):
        return 'LinearBlock(in_feat: {:d}, out_feat: {:d}, BN: {:s}, ReLU: {:s})'.format(self.in_dim,
                                                                                        self.out_dim,
                                                                                        str(self.use_bn),
                                                                                        str(not self.no_relu))
    
    
class LinearBlock_V2(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn, bn_momentum, no_relu=False):
        super(LinearBlock_V2, self).__init__()
        self.bn_momentum = bn_momentum
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.no_relu = no_relu
        self.mlp = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim, momentum=self.bn_momentum)
        
    def forward(self, x):
        x = self.mlp(x)
        x = x.transpose(1, 2)
        x = self.bn(x)
        x = x.transpose(1, 2)
        
        return x

    
class ResNetBlock(nn.Module):
    def __init__(self, block_name, in_dim, out_dim, radius, layer_ind, config):
        super(ResNetBlock, self).__init__()
        
        # Get KP_extent from current radius
        current_extent = radius * config.KP_extent / config.conv_radius
        
        # Get other params
        self.bn_momentum = config.batch_norm_momentum
        self.use_bn = config.use_batch_norm
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.radius = radius
        self.kernel_size = config.num_kernel_points
        self.block_name = block_name
        self.layer_ind = layer_ind
        
        # First downscaling mlp
        if in_dim != out_dim // 4:
            self.unary1 = LinearBlock(in_dim, out_dim // 4, self.use_bn, self.bn_momentum)
        else:
            self.unary1 = nn.Identity()
            
        self.pos_encoding = PosEncoding(out_dim // 4, out_dim // 4, self.use_bn, self.bn_momentum)
            
        # KPConv block
        self.ConvOP1 = ConvOP(self.kernel_size,
                            config.in_points_dim,
                            out_dim // 4,
                            out_dim // 4,
                            current_extent,
                            radius,
                            fixed_kernel_points=config.fixed_kernel_points,
                            KP_influence=config.KP_influence,
                            aggregation_mode=config.aggregation_mode,
                            deformable=False,
                            modulated=False)
        
        
        self.ConvOP2 = ConvOP(self.kernel_size - 5,
                            config.in_points_dim,
                            out_dim // 4,
                            out_dim // 4,
                            current_extent,
                            radius*0.75,
                            fixed_kernel_points=config.fixed_kernel_points,
                            KP_influence=config.KP_influence,
                            aggregation_mode=config.aggregation_mode,
                            deformable=False,
                            modulated=False)
        
        
        self.ConvOP3 = ConvOP(15,
                             config.in_points_dim,
                             out_dim // 2,
                             out_dim // 4,
                             current_extent,
                             radius,
                             fixed_kernel_points=config.fixed_kernel_points,
                             KP_influence=config.KP_influence,
                             aggregation_mode=config.aggregation_mode,
                             deformable=False,
                             modulated=False)
        
        self.batch_norm_conv = BatchNormBlock(out_dim//4, self.use_bn, self.bn_momentum)
        self.batch_norm_attention = BatchNormBlock(out_dim//4, self.use_bn, self.bn_momentum)
     
        
        self.attentive_head = AttentionHead(out_dim // 4, out_dim // 4, self.use_bn, self.bn_momentum)
        self.attentive_head_global = GlobalAttentionHead(out_dim // 4, out_dim // 4, self.use_bn, 
                                                        self.bn_momentum)
        
        
        self.attentive_unary = LinearBlock(out_dim // 4 + out_dim // 4, out_dim // 4, self.use_bn, self.bn_momentum)
        
        
        self.unary2 = LinearBlock(out_dim // 4, out_dim, self.use_bn, self.bn_momentum, no_relu=True)
        
        # Shortcut optional mlp
        if in_dim != out_dim:
            self.unary_shortcut = LinearBlock(in_dim, out_dim, self.use_bn, self.bn_momentum)
        else:
            self.unary_shortcut = nn.Identity()
            
            
        self.adaptive_pool = torch.nn.AdaptiveMaxPool1d(out_dim // 4)
        
        # Other ops
        self.leaky_relu = nn.LeakyReLU(0.1)
        
        return
    
    
    def forward(self, features, batch):
        if 'strided' in self.block_name:
            q_points = batch.points[self.layer_ind + 1]
            s_points = batch.points[self.layer_ind]
            neighbor_indices = batch.pools[self.layer_ind]
        else:
            q_points = batch.points[self.layer_ind]
            s_points = batch.points[self.layer_ind]
            neighbor_indices = batch.neighbors[self.layer_ind]

                
        residual = self.unary1(features)
       
        
        if not 'strided' in self.block_name:
#             pos_residual = self.pos_encoding(s_points, neighbor_indices, residual)
            conv_residual1 = self.ConvOP1(q_points, s_points, neighbor_indices, residual)
#             conv_residual1 = self.leaky_relu(self.batch_norm_conv(conv_residual1))
            conv_residual2 = self.ConvOP2(q_points, s_points, neighbor_indices, residual)
#             conv_residual2 = self.leaky_relu(self.batch_norm_conv(conv_residual2))
            conv_residual = conv_residual1 + conv_residual2
            
            if 'final' in self.block_name:
                attention_residual = self.attentive_head_global(conv_residual)
            else:
                attention_residual = self.attentive_head(s_points, neighbor_indices, conv_residual)
#             attention_residual = self.attentive_head_global(residual)
#             attention_residual = self.leaky_relu(self.batch_norm_conv(attention_residual))
            
#             residual = conv_residual + attention_residual
            residual = torch.cat([attention_residual, conv_residual], dim=-1)
    
            residual = self.adaptive_pool(residual.unsqueeze(0)).squeeze(0)
        
            residual = self.leaky_relu(self.batch_norm_conv(residual))
            
        else:
            residual = self.ConvOP1(q_points, s_points, neighbor_indices, residual)
            residual = self.leaky_relu(self.batch_norm_conv(residual))
            
        
        
        residual = self.unary2(residual)
        
        
        if 'strided' in self.block_name:
            shortcut = max_pool(features, neighbor_indices)
            
        else:
            shortcut = features
            
        shortcut = self.unary_shortcut(shortcut)
        
        q_feats = residual + shortcut
        q_feats = self.leaky_relu(q_feats)
        
        return q_feats
    
    
class NearestUpsampleBlock(nn.Module):
    def __init__(self, layer_ind):
        """
        Initialize nearest upsampling block.
        """
        
        super(NearestUpsampleBlock, self).__init__()
        self.layer_ind = layer_ind
        return
    
    def forward(self, x, batch):
        return closest_pool(x, batch.upsamples[self.layer_ind - 1])
