from torch import nn
import torch

from model.utils import get_activation
from model.vit import ViTEncoder

class MaskedFeature(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, step):
        mask = torch.zeros(x.shape).to(x.device)
        mask[:, :step+1, :] = 1
        x = x * mask
        return x

class ViTAutoRegressor(nn.Module):
    def __init__(self, 
                 dim, 
                 depth, 
                 heads, 
                 mlp_dim, 
                 num_features, 
                 num_classes,
                 patch_size, 
                 channels = 1, 
                 dim_head = 64,
                 activation = 'relu',
                 device = 'cuda'):
        super().__init__()

        self.vit_encoder = ViTEncoder(dim, 
                                      depth, 
                                      heads, 
                                      mlp_dim, 
                                      num_features+1+2, 
                                      patch_size, 
                                      channels, 
                                      dim_head,
                                      activation,
                                      device)
        
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

    def forward(self, x):

        x = self.vit_encoder(x)
        x = x.mean(dim=1)

        lith_output = self.lithology_head(x)
        lith_output_prob = lith_output.softmax(dim=-1)

        x_lith = torch.cat((x, lith_output_prob), dim=-1)
        phi_output = self.phi_head(x_lith)

        x_lith_phi = torch.cat((x, lith_output_prob, phi_output), dim=-1)
        sw_output = self.sw_head(x_lith_phi).sigmoid()

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

    return ViTAutoRegressor(
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
    )