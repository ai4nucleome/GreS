import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LLMHead(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim_ratio=0.5):
        super().__init__()

        hidden_dim = max(1, int(input_dim * hidden_dim_ratio))
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.mlp(x)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

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
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCN(nn.Module):
    """GCN encoder with two graph convolution layers."""
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


class ZINBDecoder(nn.Module):
    """ZINB decoder for gene expression reconstruction."""
    
    def __init__(self, nfeat, nhid1, nhid2):
        super(ZINBDecoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.BatchNorm1d(nhid1),
            nn.ReLU()
        )
        self.pi = nn.Linear(nhid1, nfeat)
        self.disp = nn.Linear(nhid1, nfeat)
        self.mean = nn.Linear(nhid1, nfeat)
        self.DispAct = lambda x: torch.clamp(F.softplus(x), 1e-4, 1e4)
        self.MeanAct = lambda x: torch.clamp(torch.exp(x), 1e-5, 1e6)

    def forward(self, emb):
        x = self.decoder(emb)
        pi = torch.sigmoid(self.pi(x))
        disp = self.DispAct(self.disp(x))
        mean = self.MeanAct(self.mean(x))
        return [pi, disp, mean]


class GreS(nn.Module):
    """
    GreS: Graph-based Spatial Transcriptomics Model with Semantic Embedding Modulation.
    
    Architecture:
    - Dual GCN encoders: spatial GCN (SGCN) and feature GCN (FGCN)
    - Gated fusion with FiLM conditioning on both gate and fused representation
    - ZINB reconstruction decoder
    """
    
    def __init__(self, nfeat, nhid1, nhid2, dropout, llm_dim, llm_modulation_ratio, use_llm=True):
        super(GreS, self).__init__()

        # GCN encoders
        self.SGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.FGCN = GCN(nfeat, nhid1, nhid2, dropout)
        self.dropout = dropout
        self.nfeat = nfeat
        self.use_llm = use_llm

        if use_llm:
            self.llm_head = LLMHead(llm_dim, nhid2)

            # FiLM modulation on fused representation
            mod_mlp_input_dim = nhid2
            mod_mlp_output_dim = 2 * nhid2  # alpha, beta each in R^{nhid2}
            mod_mlp_hidden_dim = max(1, int(mod_mlp_input_dim * llm_modulation_ratio))
            self.modulation_mlp = nn.Sequential(
                nn.Linear(mod_mlp_input_dim, mod_mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(mod_mlp_hidden_dim, mod_mlp_output_dim)
            )

            # FiLM modulation on gate input
            gate_in_dim = 2 * nhid2
            gate_mod_out_dim = 2 * gate_in_dim  # alpha_gate, beta_gate each in R^{2*nhid2}
            gate_mod_hidden_dim = max(1, int(gate_in_dim * llm_modulation_ratio))
            self.gate_modulation_mlp = nn.Sequential(
                nn.Linear(nhid2, gate_mod_hidden_dim),  # input is processed_llm_emb (nhid2)
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(gate_mod_hidden_dim, gate_mod_out_dim),
            )

        # Gated fusion:
        # gate = sigmoid(MLP([emb1, emb2])) in R^{nhid2}
        # fused = gate * emb1 + (1-gate) * emb2
        self.gate_mlp = nn.Sequential(
            nn.Linear(nhid2 * 2, nhid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid1, nhid2),
            nn.Sigmoid(),
        )

        # Post-fusion projection
        self.post_fusion_mlp = nn.Sequential(
            nn.Linear(nhid2, nhid1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(nhid1, nhid2),
        )

        # ZINB decoder
        self.ZINB = ZINBDecoder(nfeat, nhid1, nhid2)

    def forward(self, x, sadj, fadj, llm_emb):
        """
        Forward pass with GCN encoders.
        
        Args:
            x: input features (normalized/scaled), shape (n_spots, nfeat)
            sadj: spatial adjacency (torch sparse)
            fadj: feature adjacency (torch sparse)
            llm_emb: LLM/semantic embeddings, shape (n_spots, llm_dim)
            
        Returns:
            latent_emb: fused embedding after post-fusion MLP
            pi: dropout probability for ZINB
            disp: dispersion for ZINB  
            mean: mean for ZINB
            emb1: spatial GCN output (for dicr_loss)
            emb2: feature GCN output (for dicr_loss)
        """
        # GCN encoders (using sparse adjacency)
        emb1 = self.SGCN(x, sadj)
        emb2 = self.FGCN(x, fadj)

        # Precompute processed LLM embedding
        processed_llm_emb = None
        if self.use_llm:
            processed_llm_emb = self.llm_head(llm_emb)  # (n_spots, nhid2)

        # Gated fusion with FiLM conditioning on gate input
        gate_in = torch.cat((emb1, emb2), dim=1)  # (n_spots, 2*nhid2)
        if self.use_llm:
            gate_params = self.gate_modulation_mlp(processed_llm_emb)  # (n_spots, 2*(2*nhid2))
            alpha_g, beta_g = torch.chunk(gate_params, 2, dim=1)       # each (n_spots, 2*nhid2)
            gate_in = (1.0 + alpha_g) * gate_in + beta_g

        gate = self.gate_mlp(gate_in)  # (n_spots, nhid2) in (0,1)
        fused = gate * emb1 + (1.0 - gate) * emb2

        # FiLM modulation on fused representation
        if self.use_llm:
            modulation_params = self.modulation_mlp(processed_llm_emb)  # (n_spots, 2*nhid2)
            alpha, beta = torch.chunk(modulation_params, 2, dim=1)      # each (n_spots, nhid2)
            fused = (1.0 + alpha) * fused + beta

        # Post-fusion projection
        fused = self.post_fusion_mlp(fused)

        # ZINB decoding
        [pi, disp, mean] = self.ZINB(fused)
        return fused, pi, disp, mean, emb1, emb2
