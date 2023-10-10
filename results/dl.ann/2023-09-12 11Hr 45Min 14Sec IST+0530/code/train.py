import hydra
import os
import shutil
import numpy as np
from time import time
from os.path import join as pjoin
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from datasets import data_preparation
from utils.misc import (add_to_config, 
                        current_time, 
                        copy_running_code, 
                        plot_all_losses_and_accuracies_curve, 
                        plot_confusion_matrix, 
                        save_best_checkpoint, 
                        save_hparams,
                        load_lora,
                        heatmap_precision_recall_f1_score,
                        load_model_from_checkpoint
                        )

def train_deep_learning(
        config, 
        x_train, 
        y_train, 
        x_val, 
        y_val, 
        num_classes
    ):
    # load the config
    data_config = config['data']
    trainer_config = config['trainer']
    random_state = config['random_state']
    model_config = config['model']
    callbacks_config = config['callbacks']

    # check which model to use
    if model_config['__name__'].endswith('vit'):
        if model_config['autoregressive']:
            from model.vit_autoregressor import build_model
            from engine.autoregressor import train
        else:
            from model.vit import build_model
            from engine.vit import train
    else:
        from model.ann import build_model
        from engine.vit import train

    #set random seed
    torch.manual_seed(random_state)
    np.random.seed(random_state)

    train_dataset = TensorDataset(x_train, 
                                  y_train)
    train_loader = DataLoader(train_dataset, 
                              batch_size=trainer_config['batch_size'], 
                              shuffle=True)

    val_dataset = TensorDataset(x_val, 
                                y_val)
    val_loader = DataLoader(val_dataset, 
                            batch_size=trainer_config['batch_size'], 
                            shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Define the loss function and optimizer
    regression_criterion = torch.nn.MSELoss()
    classification_criterion = nn.CrossEntropyLoss(weight=torch.tensor(data_config['class_weights']).float().to(device))
    loss_weights = model_config['loss_weights']

    # Training loop
    num_epochs = trainer_config['epochs']

    # add new key to hydra config
    add_to_config(data_config, 'num_features', x_train.shape[-1])
    add_to_config(trainer_config, 'device', str(device))

    model = build_model(config)
    if config['trainer']['policy_exp_path']:
        model, _ = load_model_from_checkpoint(model, 'checkpoint.pt', config['trainer']['policy_exp_path'], device)
    if config['model']['use_lora']:
        model = load_lora(model, config, 'checkpoint.pt', device)
    model.to(device)

    if trainer_config['optim'] == 'adam':
        print("Using Adam optimizer")
        optimizer = optim.Adam(model.parameters(), lr=trainer_config['lr'])
    elif trainer_config['optim'] == 'sgd':
        print("Using SGD optimizer")
        optimizer = optim.SGD(model.parameters(), lr=trainer_config['lr'], momentum=0.9)
    elif trainer_config['optim'] == 'adamw':
        print("Using AdamW optimizer")
        optimizer = optim.AdamW(model.parameters(), lr=trainer_config['lr'])
    else:
        raise ValueError(f"Invalid optimizer: {trainer_config['optim']}. Valid options are: adam, sgd, adamw")

    print("Starting training...")
    if model_config['autoregressive']:
        (
            train_losses, 
            val_losses, 
            train_lith_losses,
            val_lith_losses,
            train_phi_losses,
            val_phi_losses,
            train_sw_losses,
            val_sw_losses,
            train_accuracies, 
            val_accuracies, 
            best_epoch, 
            best_loss,
            best_accuracy,
            best_cm_val,
            best_cm,
            model_chkpt,
            optim_chkpt
        ) = train(
            num_epochs, 
            model, 
            train_loader, 
            val_loader, 
            regression_criterion,
            classification_criterion,
            loss_weights,
            optimizer,
            callbacks_config['early_stopping_tolerance'],
            device,
            data_config['patch']['patch_size'],
            num_classes
        )
    else:
        (
            train_losses, 
            val_losses, 
            train_lith_losses,
            val_lith_losses,
            train_phi_losses,
            val_phi_losses,
            train_sw_losses,
            val_sw_losses,
            train_accuracies, 
            val_accuracies, 
            best_epoch, 
            best_loss,
            best_accuracy,
            best_cm_val,
            best_cm,
            model_chkpt,
            optim_chkpt
        ) = train(
            num_epochs, 
            model, 
            train_loader, 
            val_loader, 
            regression_criterion,
            classification_criterion,
            loss_weights,
            optimizer,
            callbacks_config['early_stopping_tolerance'],
            device,
        )

    print(f"Best epoch: {best_epoch+1}, Best loss: {best_loss:.4f}, Best accuracy: {best_accuracy:.2f}%")

    print("Plotting loss and accuracy curves...")
    plot_all_losses_and_accuracies_curve(train_losses,
                                         val_losses, 
                                         train_lith_losses, 
                                         val_lith_losses, 
                                         train_phi_losses, 
                                         val_phi_losses,
                                         train_sw_losses, 
                                         val_sw_losses,
                                         train_accuracies, 
                                         val_accuracies,
                                         trainer_config['experiment_path'])
    plot_confusion_matrix(best_cm, 
                          best_cm_val, 
                          data_config['lithology_classes'], 
                          trainer_config['experiment_path'])
    heatmap_precision_recall_f1_score(best_cm, 
                                      best_cm_val, 
                                      data_config['lithology_classes'], 
                                      trainer_config['experiment_path'])

    save_best_checkpoint(
        model_chkpt, 
        optim_chkpt, 
        train_losses, 
        val_losses, 
        train_accuracies, 
        val_accuracies, 
        best_epoch, 
        best_loss, 
        trainer_config
        )
    save_hparams(config, trainer_config)

@hydra.main(config_path="config", config_name="defaults", version_base='1.3.2')
def main(config: DictConfig):
    start = time()

    save_path = pjoin(config['trainer']['save_dir'], 
                        config['model']['__name__'], 
                        current_time())
    code_save_path = pjoin(save_path, 'code')
    copy_running_code(code_save_path)
    add_to_config(config['trainer'], 'experiment_path', save_path)
    os.makedirs(pjoin(save_path, config['trainer']['scaler_path']), exist_ok=True)
    
    # load the dataset
    (
        x_train, 
        x_val, 
        y_train, 
        y_val, 
        num_classes 
    ) = data_preparation.prepare_data(
        config
    )

    # check if the model is classical ML or deep learning
    if config['model']['__name__'].startswith('ml'):
        from engine.classical_ml import train_classical_ml

        print("Training classical ML model...")
        train_classical_ml(config, 
                           x_train, 
                           y_train, 
                           x_val, 
                           y_val)
    else:
        print("Training deep learning model...")

        train_deep_learning(
            config, 
            x_train, 
            y_train, 
            x_val, 
            y_val,
            num_classes
        )
    stop = time()
    print("Training complete!")
    print(f"Total time taken: {(stop-start)/60:.2f} minutes")
    shutil.move('output.log', pjoin(save_path, 'output.log'))
if __name__ == '__main__':
    main()
