from torch import nn
import torch

from einops import rearrange
from einops.layers.torch import Rearrange

from model.transformers import Transformer, posemb_sincos_2d
from model.utils import get_activation

class ViTEncoder(nn.Module):
    def __init__(self,
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 num_features, 
                 patch_size, 
                 channels = 1, 
                 dim_head = 64,
                 activation = 'relu',
                 device = 'cuda',
                 return_attn = False):
        super().__init__()

        image_height, image_width = patch_size, num_features
        patch_height, patch_width = 1, num_features
        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch dimentions.'

        patch_dim = channels * patch_height * patch_width

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (h p1) (w p2) -> b h w (p1 p2)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )

        if return_attn:
            self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, activation, return_attn = return_attn)
        else:
            encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # self.to_latent = nn.Identity()
        self.to_latent = nn.Sequential(
            nn.LayerNorm(dim),
            get_activation(activation),
            nn.Linear(dim, dim)
        )

        self.device = device
        self.return_attn = return_attn

    def forward(self, img):

        x = self.to_patch_embedding(img)
        pe = posemb_sincos_2d(x).to(self.device)
        x = rearrange(x, 'b ... d -> b (...) d') + pe

        if self.return_attn:
            x, att = self.transformer(x)
        else:
            x = self.transformer(x)
        # x = x.mean(dim = 1)

        x = self.to_latent(x)
        if self.return_attn:
            return x, att
        else:
            return x
    
class SimpleViT(nn.Module):
    def __init__(self, 
                 num_classes, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 num_features, 
                 patch_size, 
                 channels = 1, 
                 dim_head = 64,
                 activation = 'relu',
                 device = 'cuda',
                 return_attn = False):
        
        super().__init__()

        self.vit_encoder = ViTEncoder(dim, depth, heads, mlp_dim, num_features, patch_size, channels, dim_head,activation,device, return_attn)

        self.sw_head = nn.Sequential(
            nn.LayerNorm(dim+num_classes+1),
            get_activation(activation),
            nn.Linear(dim+num_classes+1, dim//2),
            nn.LayerNorm(dim//2),
            get_activation(activation),
            nn.Linear(dim//2, 1)
        )

        self.lithology_head = nn.Sequential(
            nn.LayerNorm(dim),
            get_activation(activation),
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            get_activation(activation),
            nn.Linear(dim//2, num_classes)
        )

        self.phi_head = nn.Sequential(
            nn.LayerNorm(dim+num_classes),
            get_activation(activation),
            nn.Linear(dim+num_classes, dim//2),
            nn.LayerNorm(dim//2),
            get_activation(activation),
            nn.Linear(dim//2, 1)
        )

        self.device = device
        self.return_attn = return_attn

    def forward(self, img): 
        if self.return_attn:
            x, att = self.vit_encoder(img)
        else:   
            x = self.vit_encoder(img)

        lith_output = self.lithology_head(x)
        lith_output_prob = lith_output.softmax(dim=-1)
        x_lith = torch.cat((x, lith_output_prob), dim=-1)
        phi_output = self.phi_head(x_lith)

        x_lith_phi = torch.cat((x, lith_output_prob, phi_output), dim=-1)
        sw_output = self.sw_head(x_lith_phi).sigmoid()

        if self.return_attn:
            return lith_output, phi_output, sw_output, att
        else:
            return lith_output, phi_output, sw_output
    
def build_model(config):
    print("Building the model...")
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']

    num_classes = len(data_config['lithology_classes'])
    num_features = data_config['num_features']
    patch_size = data_config['patch']['patch_size']
    activation = model_config['activation']

    return SimpleViT(
           num_classes = num_classes, 
           dim = model_config['dim'],
           depth = model_config['depth'], 
           heads = model_config['heads'], 
           mlp_dim = model_config['mlp_dim'], 
           num_features = num_features, 
           patch_size = patch_size,
           channels = model_config['channels'],
           dim_head = model_config['dim_head'],
           activation = activation,
           device = trainer_config['device'],
           return_attn = model_config['return_attention_map']
    )