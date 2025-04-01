import math
import torch
import torch.nn as nn
import options
from custom_types import *
from models import transformer, models_utils
from torch.nn import Parameter
from torch.hub import load as torch_hub_load
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


dinov2_vitb14 = torch_hub_load('facebookresearch/dinov2', 'dinov2_vitb14')


class GraphConvolution(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

class SketchRefinement(nn.Module):

    def forward(self, x: T):
        query = self.query_embeddings.repeat(x.shape[0], 1, 1)
        x = x + query
        out = self.encoder(x)
        return out

    def __init__(self, opt: options.SketchOptions):
        super(SketchRefinement, self).__init__()
        dim_ref = int(opt.dim_h * 1.5)
        self.encoder = transformer.Transformer(dim_ref, 8, 12)
        query_embeddings = torch.zeros(1, opt.num_gaussians, dim_ref)
        self.query_embeddings = nn.Parameter(query_embeddings)
        torch.nn.init.normal_(
            self.query_embeddings.data,
            0.0,
            1. / math.sqrt(dim_ref),
        )

class Sketch2Spaghetti(models_utils.Model):

    def get_visual_embedding_and_queries(self, x: T):
        
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)         # 224
        x = x.repeat(1, 3, 1, 1)                                                            # 3 channel
        x = self.dino.get_intermediate_layers(x)[0]                                         # DINOv2 
        x = self.fc(x)                                                                      # 768 -> 512
        
        query = self.query_embeddings.repeat(x.shape[0], 1, 1)
        return x, query

    def forward_attention(self, x: T):
        x, query = self.get_visual_embedding_and_queries(x)
        out, attention = self.decoder.forward_attention(query, x)
        cls = self.mlp_cls(out)
        zh = self.mlp_zh(out)
        return zh, cls, attention

    def forward_mid(self, x, return_mid):
        cls = self.mlp_cls(x)
        zh = self.mlp_zh(x)
        if return_mid:
            return zh, cls, x
        return zh, cls
    
    def get_adj_indiv(self, query, mode='indiv'):
        query = query.clone().detach()
        if mode == 'indiv':
            adj = self.adj_predictor_indiv(query)                                               # [batch, num_gaussians, dim_ref]
        elif mode == 'part':
            adj = self.adj_predictor_part(query)
        adj = torch.bmm(adj, adj.transpose(1, 2))                                               # [batch, num_gaussians, num_gaussians]
        adj = F.softmax(adj, dim=-1)                                                            # Normalize adjacency matrix
        # fill diagonal with 1
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        adj = adj * (1 - eye) + eye
        return adj
    

    def forward(self, x: T, l_feat, return_mid=False):
        '''
            dist_adj: calculated individual adjacency by distance
        '''
        x, query = self.get_visual_embedding_and_queries(x)
        out = self.decoder(query, x, l_feat)

        # Individual Graph Convolution
        indiv_adj = self.get_adj_indiv(query.clone().detach(), mode='indiv')

        adj_indivisual = indiv_adj

        query_indivisual1 = F.relu(self.gcn1(out, adj_indivisual)) 
        query_indivisual2 = F.relu(self.gcn2(query_indivisual1, adj_indivisual))
        
        grouped_avg = torch.zeros(query.shape[0], 4, query.shape[-1], device=x.device)

        all_indices = []
        for bi in range(indiv_adj.shape[0]):
            adj = indiv_adj[bi].detach().cpu().numpy()
            Z = linkage(adj, 'average')
            clusters = fcluster(Z, 4, criterion='maxclust')
            all_indices.append(clusters - 1)

        all_indices = torch.tensor(all_indices, device=x.device, dtype=torch.long)
        all_indices_exp = all_indices.unsqueeze(-1).expand(-1, -1, query.shape[-1])

        grouped_avg = torch.scatter_reduce(
            grouped_avg, 1, all_indices_exp, out, reduce='mean', include_self=False
        )
        part_adj = self.get_adj_indiv(grouped_avg, mode='part')
        
        query_parts1=F.relu(self.gcn1(grouped_avg, part_adj))
        query_parts2=F.relu(self.gcn2(query_parts1, part_adj))
        query_parts3 = query_parts2.gather(dim=1, index=all_indices_exp)
        gcn_out = query_indivisual2* 0.8  + query_parts3 * 0.2
        
        out = out + gcn_out
        out = self.out_norm(out)
        return self.forward_mid(out, return_mid), indiv_adj, part_adj

    def refine(self, mid_embedding, return_mid=False):
        out = self.refinement_encoder(mid_embedding)
        zh = self.mlp_zh(out)
        if return_mid:
            return zh, out
        return zh

    def __init__(self, opt: options.SketchOptions):
        super(Sketch2Spaghetti, self).__init__()
        self.opt = opt
        
        self.dino = dinov2_vitb14
        self.fc = nn.Linear(768, 512)

        
        dim_ref = int(opt.dim_h * 1.5)
        
        dim_llava = 4096
        
        self.decoder = transformer.CombLlavaTransformer(dim_ref, dim_llava, 8, 12, opt.dim_h)
        query_embeddings = torch.zeros(1, opt.num_gaussians, dim_ref)
        self.query_embeddings = nn.Parameter(query_embeddings)
        torch.nn.init.normal_(
            self.query_embeddings.data,
            0.0,
            1. / math.sqrt(dim_ref),
        )
        self.mlp_zh = models_utils.MLP((dim_ref, opt.dim_h, opt.dim_h), norm_class=None)
        self.mlp_cls = models_utils.MLP((dim_ref, opt.dim_h, 1), norm_class=None)
        
        if opt.refinement:
            self.refinement_encoder = SketchRefinement(opt)
        
        self.gcn1 = GraphConvolution(dim_ref, dim_ref)
        self.gcn2 = GraphConvolution(dim_ref, dim_ref)

        self.adj_predictor_indiv = nn.Sequential(
            nn.Linear(dim_ref, dim_ref),
            nn.ReLU(),
            nn.Linear(dim_ref, dim_ref),
        )
        self.adj_predictor_part = nn.Sequential(
            nn.Linear(dim_ref, dim_ref),
            nn.ReLU(),
            nn.Linear(dim_ref, dim_ref),
        )
        self.out_norm = nn.LayerNorm(dim_ref)

def main():
    model = Sketch2Spaghetti(options.SketchOptions())
    x = torch.rand(5, 1, 256, 256)
    out = model(x)
    print(out.shape)

if __name__ == '__main__':
    main()
