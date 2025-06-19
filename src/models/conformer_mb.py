"""
Input: GOP feature x in shape [batch_size, seq_len, feat_dim], 
e.g., [25, 50, 84] for a batch of 25 utterances, each with 50 phones after -1 padding, 
and each phone has a GOP feature vector of dimension 84.

Input: canonical phone phn in shape [batch_size, seq_len, phn_num], 
e.g., [25, 50, 40] for a batch of 25 utterance, 
each with 50 phones after padding with a phone dictionary of size of 40. 
For speechocean762, phn_num=40.

Output: Tuple of [u1, u2, u3, u4, u5, p, w1, w2, w3] 
where u{1-5} are utterance-level scores in shape [batch_size, 1]; 
p and w{1-3} are phone-level and word-level score in shape [batch_size, seq_len]. 
Note we propagate word score to phone-level, so word output should also be at phone-level.
"""
import math
import torch
from torch import nn
from torchrl.modules import TruncatedNormal
from torchaudio.models import Conformer

class PositionalEncoding(nn.Module):
    """
    from pytorch's Transformer tutorial 
    https://pytorch-tutorials-preview.netlify.app/beginner/transformer_tutorial.html
    modified to be initialized with truncated normal distribution
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        #position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        position = torch.empty(size=(max_len, 1)) # [seq_len, batch_size]
        position = nn.init.trunc_normal_(position) # TODO adjust parameters
        """position = TruncatedNormal(
            # means
            loc = torch.zeros(d_model), # TODO : test different means
            # STDs
            scale = torch.ones(d_model), # TODO : test different STDs
            # truncation parameters # TODO
            low = -5, # looking at 5 past tokens ?
            high = 0 # only looking at past tokens
            )"""

        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term) # one on two values is sin(position(value) * div_term), starting from 0
        pe[:, 0, 1::2] = torch.cos(position * div_term) # so the other values are cos(position(value) * div_term), so starting from 1
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        # add positional embedding to the tensor
        x = x + self.pe[:x.size(0)] # type: ignore
        return self.dropout(x)
    

class ConformerMB(nn.Module) :
    def __init__(
            self,
            embed_dim:int = 512,
            num_heads:int=1,
            depth:int = 3,
            input_dim:int = 84) :
        super().__init__()
        # Here I use the same layer and parameter names as the ConformerMB paper,
        # or when the name is missing, from the GOPT github repo

        # each phone has a GOP feature vector of dimension 84
        self.feat_dim = input_dim
        # dimension of the GOP and phoneme embeddings
        self.embedding_dim = embed_dim
        # number of possible phonemes
        self.phn_num = 40

        # linear layer to map the pronunciation goodness feature dimension 
        # to the same dimension, embedding_dim, as that of the text embedding layer
        self.gop_projection_layer = nn.Linear(84, self.embedding_dim)

        # encode the different phonemes in the audio text sequence 
        # to obtain the text content features and encode the position of the phonemes 
        # to get the position of different phonemes in different text sequences
        self.phoneme_embedding = nn.Embedding(self.phn_num, self.embedding_dim)

        # initialize the position parameters of the reference text sequence, 
        # resulting in a text position encoding
        self.positional_embedding = PositionalEncoding(d_model=self.embedding_dim) # TODO : adjust/remove dropout
        
        # 
        self.conformer_encoder = Conformer(
            input_dim=input_dim, 
            num_heads=num_heads,
            num_layers=depth,
            ffn_dim=self.embedding_dim,
            depthwise_conv_kernel_size=31
            )
        
        # utterance scores
        self.acc_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.fluency_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.completeness_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.prosodic_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.total_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        
        # phoneme and word-level scores
        self.phoneme_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.word_acc_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.word_stress_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )
        self.word_total_head = nn.Sequential(
            nn.Linear(self.embedding_dim, 1),
            nn.Sigmoid()
        )

        

    def forward(
        self,
        x:torch.FloatTensor,
        phn:torch.IntTensor|torch.LongTensor|None = None,
    ):
        projected_gop = self.gop_projection_layer(x)
        batch_size = projected_gop.shape[0]
        seq_len = projected_gop.shape[1]

        if phn is not None :
            phonemes = torch.max(phn, dim=2).indices # type: ignore
            embedded_phn = self.phoneme_embedding(phonemes)
            positions = self.positional_embedding(embedded_phn.permute(1,0,2)) # type: ignore # permute sequence and batch

            h = projected_gop + positions.permute(1,0,2) # embedded_phn and their positional encoding were already added, if this is what the paper meant ?
            o = self.conformer_encoder(
                input = h,
                lengths = torch.full(size=(batch_size,), fill_value=seq_len) # TODO here we consider that the whole sequence length is valid, is it true tho ?
                )

            #return (projected_gop, embedded_phn, positions.permute(1,0,2), h, o)

            u1 = self.acc_head(o[0])
            u2 = self.fluency_head(o[0])
            u3 = self.completeness_head(o[0])
            u4 = self.prosodic_head(o[0])
            u5 = self.total_head(o[0])
            p = self.phoneme_head(o[0])
            w1 = self.word_acc_head(o[0])
            w2 = self.word_stress_head(o[0])
            w3 = self.word_total_head(o[0])

            return (u1, u2, u3, u4, u5, p, w1, w2, w3)

