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

class PredAutoRegressor(nn.Module):
    def __init__(self,  
                 dim,
                 num_classes, 
                 activation,
                 auto_regressor_hidden_layer_sizes):
        super().__init__()
        assert auto_regressor_hidden_layer_sizes[-1] == dim, "Last layer of autoregressor encoder must have same size as last layer of vit encoder"
        auto_regressor_enc_layers = []
        auto_regressor_layer_sizes = [num_classes] + auto_regressor_hidden_layer_sizes
        
        for i in range(len(auto_regressor_layer_sizes) - 1):
            auto_regressor_enc_layers.append(nn.Linear(auto_regressor_layer_sizes[i], auto_regressor_layer_sizes[i+1]))
            auto_regressor_enc_layers.append(nn.LayerNorm(auto_regressor_layer_sizes[i+1]))
            auto_regressor_enc_layers.append(get_activation(activation))
            auto_regressor_enc_layers.append(MaskedFeature())

        self.auto_regressor = nn.Sequential(*auto_regressor_enc_layers)
        
    def forward(self, x, step):
        for layer in self.auto_regressor:
            if isinstance(layer, MaskedFeature):
                x = layer(x, step)
            else:
                x = layer(x)
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
                 auto_regressor_hidden_layer_sizes,
                 channels = 1, 
                 dim_head = 64,
                 activation = 'relu',
                 device = 'cuda'):
        super().__init__()

        self.vit_encoder = ViTEncoder(dim, 
                                      depth, 
                                      heads, 
                                      mlp_dim, 
                                      num_features, 
                                      patch_size, 
                                      channels, 
                                      dim_head,
                                      activation,
                                      device)
        
        # self.rnn = nn.LSTM(dim, dim, 1, batch_first=True)
        
        self.auto_regressor_ = PredAutoRegressor(dim, 
                                                num_classes+2, 
                                                activation,
                                                auto_regressor_hidden_layer_sizes)
        
        self.merge_layer = nn.Sequential(
            nn.Linear(dim, dim//2),
            nn.LayerNorm(dim//2),
            get_activation(activation),
        )
        
        self.sw_head = nn.Sequential(
            nn.LayerNorm((dim//2)+num_classes+1),
            get_activation(activation),
            nn.Linear((dim//2)+num_classes+1, (dim//2)//2),
            nn.LayerNorm((dim//2)//2),
            get_activation(activation),
            nn.Linear((dim//2)//2, 1)
        )

        self.lithology_head = nn.Sequential(
            nn.LayerNorm((dim//2)),
            get_activation(activation),
            nn.Linear((dim//2), (dim//2)//2),
            nn.LayerNorm((dim//2)//2),
            get_activation(activation),
            nn.Linear((dim//2)//2, num_classes)
        )

        self.phi_head = nn.Sequential(
            nn.LayerNorm((dim//2)+num_classes),
            get_activation(activation),
            nn.Linear((dim//2)+num_classes, (dim//2)//2),
            nn.LayerNorm((dim//2)//2),
            get_activation(activation),
            nn.Linear((dim//2)//2, 1)
        )

    def forward(self, x, y, step, vit_emdedding = None):
        if vit_emdedding is None:
            vit_emdedding = self.vit_encoder(x)
        y = self.auto_regressor_(y, step)
        x = vit_emdedding * y
        x = x[:, :step+1, :].mean(dim=1)
        x = self.merge_layer(x)

        lith_output = self.lithology_head(x)
        lith_output_prob = lith_output.softmax(dim=-1)

        x_lith = torch.cat((x, lith_output_prob), dim=-1)
        phi_output = self.phi_head(x_lith)

        x_lith_phi = torch.cat((x, lith_output_prob, phi_output), dim=-1)
        sw_output = self.sw_head(x_lith_phi).sigmoid()

        return lith_output, phi_output, sw_output, vit_emdedding
    
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
           auto_regressor_hidden_layer_sizes = model_config['auto_regressor_hidden_layer_sizes'],
           num_features = num_features, 
           patch_size = patch_size,
           channels = model_config['channels'],
           dim_head = model_config['dim_head'],
           activation = activation,
           device = trainer_config['device'],
    )