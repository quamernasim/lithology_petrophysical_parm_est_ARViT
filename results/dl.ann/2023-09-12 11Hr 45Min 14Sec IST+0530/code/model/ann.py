from torch import nn
import torch
from model.utils import get_activation

class ANN(nn.Module):
    def __init__(self, 
                 num_features, 
                 num_classes, 
                 activation='relu',
                 norm_dim = None,
                 hidden_layer_sizes=[8, 4, 2]):
        
        super(ANN, self).__init__()

        self.norm_dim = norm_dim

        # feature encoder
        layers = []
        layer_sizes = [num_features] + hidden_layer_sizes + [num_classes]

        for i in range(len(layer_sizes) - 2):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
            if self.norm_dim:
                layers.append(nn.BatchNorm1d(self.norm_dim))
            layers.append(get_activation(activation))

        self.encoder = nn.Sequential(*layers)

        dim = layer_sizes[-2]

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

    def forward(self, x, pred_prob=None):
        x = self.encoder(x)
        
        lith_output = self.lithology_head(x)
        lith_output_prob = lith_output.softmax(dim=-1)

        x_lith = torch.cat((x, lith_output_prob), dim=2)
        phi_output = self.phi_head(x_lith)

        x_lith_phi = torch.cat((x, lith_output_prob, phi_output), dim=2)
        sw_output = self.sw_head(x_lith_phi).sigmoid()

        return lith_output, phi_output, sw_output
    
def build_model(config):
    print("Building the model...")
    data_config = config['data']
    model_config = config['model']
    trainer_config = config['trainer']

    num_classes = len(data_config['lithology_classes'])
    num_features = data_config['num_features']

    activation = model_config['activation']

    norm_dim = data_config['patch']['patch_size'] if model_config['use_batch_norm'] else None

    return ANN(
        num_features = num_features,
        num_classes = num_classes,
        activation = activation,
        norm_dim = norm_dim,
        hidden_layer_sizes = model_config['hidden_layers'],
    )
