import torch
import numpy as np
import torch.nn as nn

def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches ** 2, h * w * c // n_patches ** 2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[:, i * patch_size: (i + 1) * patch_size, j * patch_size: (j + 1) * patch_size]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = np.sin(i / (10000 ** (j / d))) if j % 2 == 0 else np.cos(i / (10000 ** ((j - 1) / d)))
    return result

class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super(MyMSA, self).__init__()
        self.d = d
        self.n_heads = n_heads

        assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"

        d_head = int(d / n_heads)
        self.q_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.k_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.v_mappings = nn.ModuleList([nn.Linear(d_head, d_head) for _ in range(self.n_heads)])
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        # Sequences has shape (N, seq_length, token_dim)
        # We go into shape    (N, seq_length, n_heads, token_dim / n_heads)
        # And come back to    (N, seq_length, item_dim)  (through concatenation)
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                seq = sequence[:, head * self.d_head: (head + 1) * self.d_head]
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                attention = self.softmax(q @ k.T / (self.d_head ** 0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
   
class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MyMSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d)
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out
   
class MyViT(nn.Module):
    def __init__(self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10):
        # Super constructor
        super(MyViT, self).__init__()
        
        # Attributes
        self.chw = chw # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        
        # Input and patches sizes
        assert chw[1] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        assert chw[2] % n_patches == 0, "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)
        
        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))
        
        # 3) Positional embedding
        self.register_buffer('positional_embeddings', get_positional_embeddings(n_patches ** 2 + 1, hidden_d), persistent=False)
        
        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList([MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])
        
        # 5) Classification MLPk
        self.mlp = nn.Linear(self.hidden_d, out_d)

    def forward(self, images):
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        
        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden size dimension
        tokens = self.linear_mapper(patches)
        
        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)
        
        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)
        
        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
            
        # Getting the classification token only
        out = out[:, 0]
        
        return self.mlp(out) # Map to output dimension, output category distribution
   

class PureWeatherModel(torch.nn.Module):
    
    def __init__(self, num_weather_channel, device, dropout=0, T_sat=12, T_wea=6) -> None:
        super().__init__()
        
     #    self.num_nonhrv_channel = 11
        self.num_weather_channel = num_weather_channel
        
        self.T_sat = T_sat
        self.T_wea = T_wea
        
        self.hrv_vit = MyViT((1, 128, 128), n_patches=8, n_blocks=2, hidden_d=32, n_heads=2, out_d=48).to(device)
        self.t_2m_vit = MyViT((1, 128, 128), n_patches=8, n_blocks=2, hidden_d=32, n_heads=2, out_d=48).to(device)
        
        # self.sat_cnn = DeepCNN(self.num_nonhrv_channel + 1, dropout=dropout)
        # self.wea_cnn = DeepCNN(self.num_weather_channel, dropout=dropout)
        
        # self.sat_pv_rnn = RNN(self.T_sat, dropout=dropout, input_size=self.sat_cnn.output_size+1, hidden_size=256, device=device)
        # self.pv_rnn = RNN(self.T_sat, dropout=dropout, input_size=1, hidden_size=128, device=device)
        # self.wea_rnn = RNN(self.T_wea, dropout=dropout, input_size=self.num_weather_channel, hidden_size=256, device=device)
        
        # output_size = self.wea_rnn.hidden_size + self.pv_rnn.hidden_size
        # output_size = self.sat_pv_rnn.hidden_size + self.wea_rnn.hidden_size

        # self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(20, 48)

    def forward(self, pv, hrv, nonhrv, weather):
        # pv:       (batch, 12)
        batch_size = pv.shape[0]
        
        # 5-minute
        # hrv:      (batch, 12, 128, 128)
        # non-hrv:  (batch, 12, 128, 128, 11)
        # sat_img:  (batch, 12, 11 + 1, 128, 128)
        # sat_img = torch.cat((hrv, nonhrv), dim=2)
        sat_img = torch.cat((hrv.unsqueeze(2), nonhrv.permute(0, 1, 4, 2, 3)), dim=2)
        
        # sat_img:  (batch * 12, 11 + 1, 128, 128)
        sat_img = sat_img.reshape(batch_size * self.T_sat, self.num_nonhrv_channel + 1, *sat_img.shape[-2:])
        
        # sat_fea:  (batch * 12, sat_cnn.output_size)
        sat_fea = self.sat_cnn(sat_img)
        
        # sat_fea_pv: (batch * 12, sat_cnn.output_size + 1)
        sat_fea_pv = torch.cat((sat_fea.reshape(batch_size, self.T_sat, sat_fea.shape[-1]), pv.unsqueeze(2)), dim=-1)

        # Hourly
        # weather:  (batch, N_weather, 6)
        # wea_img:  (batch, 6, N_weather, 128, 128)
        # wea_img = weather.transpose(1, 2)
        
        # wea_fea:  (batch * 6, N_weather, 128, 128)
        # wea_img = wea_img.reshape(batch_size * self.T_wea, self.num_weather_channel, *weather.shape[-2:])
        # wea_fea:  (batch * 6, wea_cnn.output_size)
        # wea_fea = self.wea_cnn(wea_img)
        # wea_fea:  (batch, 6, wea_cnn.output_size)
        # wea_fea = wea_fea.reshape(batch_size, self.T_wea, wea_fea.shape[-1])
        
        o_sat_pv = self.sat_pv_rnn(sat_fea_pv)
        # o_pv = self.pv_rnn(pv.unsqueeze(-1))
        o_wea = self.wea_rnn(weather.transpose(1, 2))

        x = torch.concat((o_sat_pv, o_wea), dim=-1)
        # x = torch.concat((self.flatten(sat_fea_pv), self.flatten(wea_fea)), dim=-1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))

        return x