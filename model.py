import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np
from torch import Tensor
from torch_encodings import PositionalEncoding1D, Summer
from mamba_ssm import Mamba


"""
Module: Time-encoder
"""
class TimeEncode(nn.Module):
    """
    out = linear(time_scatter): 1-->time_dims
    out = cos(out)
    """
    def __init__(self, dim):
        super(TimeEncode, self).__init__()
        self.dim = dim
        self.w = nn.Linear(1, dim)
        self.reset_parameters()
    
    def reset_parameters(self, ):
        self.w.weight = nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.dim, dtype=np.float32))).reshape(self.dim, -1))
        self.w.bias = nn.Parameter(torch.zeros(self.dim))

        self.w.weight.requires_grad = False
        self.w.bias.requires_grad = False
    
    @torch.no_grad()
    def forward(self, t):
        output = torch.cos(self.w(t.reshape((-1, 1))))
        return output


"""
Module: MambaTHN
"""

class FeatEncode(nn.Module):
    """
    Return [raw_edge_feat | TimeEncode(edge_time_stamp)]
    """
    def __init__(self, time_dims, feat_dims, out_dims):
        super().__init__()
        
        self.time_encoder = TimeEncode(time_dims)
        self.feat_encoder = nn.Linear(time_dims + feat_dims, out_dims) 
        self.reset_parameters()

    def reset_parameters(self):
        self.time_encoder.reset_parameters()
        self.feat_encoder.reset_parameters()
        
    def forward(self, edge_feats, edge_ts):
        edge_time_feats = self.time_encoder(edge_ts)
        x = torch.cat([edge_feats, edge_time_feats], dim=1)
        return self.feat_encoder(x)


class MambaBlock(nn.Module):
    """
    Semantic Patches Fusion Module with Mamba
    """
    def __init__(self, dims, dropout=0.2, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.norm = nn.RMSNorm(dims)  
        # self.norm = nn.LayerNorm(dims)      
        self.mamba = Mamba(
            d_model=dims,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.dropout = nn.Dropout(dropout)


    def reset_parameters(self):
        self.norm.reset_parameters()

    def forward(self, x):                   # x: (B, L, C)
        y = self.mamba(self.norm(x))
        return x + self.dropout(y)          


class Patch_Encoding(nn.Module):
    """
    Input : [ batch_size, graph_size, edge_dims+time_dims]
    Output: [ batch_size, graph_size, output_dims]
    """
    def __init__(self, per_graph_size, time_channels,
                 input_channels, hidden_channels, out_channels,
                 num_layers, dropout,
                 window_size,
                 module_spec=None
                ):
        super().__init__()
        self.per_graph_size = per_graph_size
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        
        # input & output classifer
        self.feat_encoder = FeatEncode(time_channels, input_channels, hidden_channels)
        # self.layernorm = nn.LayerNorm(hidden_channels)
        self.rmsnorm = nn.RMSNorm(hidden_channels)
        self.mlp_head = nn.Linear(hidden_channels, out_channels)
        
        # inner layers
        self.mixer_blocks = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.mixer_blocks.append(
                MambaBlock(hidden_channels, dropout=dropout)
            )
        # padding
        self.stride = window_size
        self.window_size = window_size
        self.pad_projector = nn.Linear(window_size*hidden_channels, hidden_channels)
        self.p_enc_1d_model_sum = Summer(PositionalEncoding1D(hidden_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mixer_blocks:
            layer.reset_parameters()
        self.feat_encoder.reset_parameters()
        self.rmsnorm.reset_parameters()
        self.mlp_head.reset_parameters()

    def forward(self, edge_feats, edge_ts, batch_size, inds):
        # x : [ batch_size, graph_size, edge_dims+time_dims]
        edge_time_feats = self.feat_encoder(edge_feats, edge_ts)
        x = torch.zeros((batch_size * self.per_graph_size, 
                         edge_time_feats.size(1)), device=edge_feats.device)
        x[inds] = x[inds] + edge_time_feats         
        x = x.view(-1, self.per_graph_size//self.window_size, self.window_size*x.shape[-1])
        x = self.pad_projector(x)
        x = self.p_enc_1d_model_sum(x) 
        for i in range(self.num_layers):
            # apply to channel + feat dim
            x = self.mixer_blocks[i](x)    
        x = self.rmsnorm(x)
        x = torch.mean(x, dim=1)
        x = self.mlp_head(x)
        return x

"""
Edge predictor
"""

class EdgePredictor_per_node(torch.nn.Module):
    """
    out = linear(src_node_feats) + linear(dst_node_feats)
    out = ReLU(out)
    """
    def __init__(self, dim_in_time, dim_in_node, predict_class):
        super().__init__()

        self.dim_in_time = dim_in_time
        self.dim_in_node = dim_in_node

        # dim_in_time + dim_in_node
        self.src_fc = torch.nn.Linear(dim_in_time+dim_in_node, 100)
        self.dst_fc = torch.nn.Linear(dim_in_time+dim_in_node, 100)
    
        self.out_fc = torch.nn.Linear(100, predict_class)
        self.reset_parameters()
        
    def reset_parameters(self, ):
        self.src_fc.reset_parameters()
        self.dst_fc.reset_parameters()
        self.out_fc.reset_parameters()

    def forward(self, h, neg_samples=1):
        num_edge = h.shape[0]//(neg_samples + 2)
        h_src = self.src_fc(h[:num_edge])
        h_pos_dst = self.dst_fc(h[num_edge:2 * num_edge])
        h_neg_dst = self.dst_fc(h[2 * num_edge:])
        
        h_pos_edge = torch.nn.functional.relu(h_src + h_pos_dst)
        h_neg_edge = torch.nn.functional.relu(h_src.tile(neg_samples, 1) + h_neg_dst)

        # h_pos_edge = torch.nn.functional.gelu(h_src + h_pos_dst)
        # h_neg_edge = torch.nn.functional.gelu(h_src.tile(neg_samples, 1) + h_neg_dst)
        
        return self.out_fc(h_pos_edge), self.out_fc(h_neg_edge)
    
    
class MambaTHN_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(STHN_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)        
        self.creterion = nn.BCEWithLogitsLoss(reduction='none') 
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, model_inputs, neg_samples, node_feats):        
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.cat((torch.ones_like(pred_pos), 
                                    torch.zeros_like(pred_neg)), dim=0)
        loss = self.creterion(all_pred, all_edge_label).mean()
        return loss, all_pred, all_edge_label
    
    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg

class Multiclass_Interface(nn.Module):
    def __init__(self, mlp_mixer_configs, edge_predictor_configs):
        super(Multiclass_Interface, self).__init__()

        self.time_feats_dim = edge_predictor_configs['dim_in_time']
        self.node_feats_dim = edge_predictor_configs['dim_in_node']

        if self.time_feats_dim > 0:
            self.base_model = Patch_Encoding(**mlp_mixer_configs)

        self.edge_predictor = EdgePredictor_per_node(**edge_predictor_configs)        
        self.creterion = nn.CrossEntropyLoss(reduction='none')
        self.reset_parameters()            

    def reset_parameters(self):
        if self.time_feats_dim > 0:
            self.base_model.reset_parameters()
        self.edge_predictor.reset_parameters()
        
    def forward(self, model_inputs, neg_samples, node_feats):        
        pos_edge_label = model_inputs[-1].view(-1,1)
        model_inputs = model_inputs[:-1]
        pred_pos, pred_neg = self.predict(model_inputs, neg_samples, node_feats)
        
        all_pred = torch.cat((pred_pos, pred_neg), dim=0)
        all_edge_label = torch.squeeze(torch.cat((pos_edge_label, torch.zeros_like(pos_edge_label)), dim=0))
        loss = self.creterion(all_pred, all_edge_label).mean()
            
        return loss, all_pred, all_edge_label
    
    def predict(self, model_inputs, neg_samples, node_feats):
        if self.time_feats_dim > 0 and self.node_feats_dim == 0:
            x = self.base_model(*model_inputs)
        elif self.time_feats_dim > 0 and self.node_feats_dim > 0:
            x = self.base_model(*model_inputs)
            x = torch.cat([x, node_feats], dim=1)
        elif self.time_feats_dim == 0 and self.node_feats_dim > 0:
            x = node_feats
        else:
            print('Either time_feats_dim or node_feats_dim must larger than 0!')
        
        pred_pos, pred_neg = self.edge_predictor(x, neg_samples=neg_samples)
        return pred_pos, pred_neg
