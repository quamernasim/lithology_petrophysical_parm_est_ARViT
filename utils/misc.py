import yaml
import datetime
import pytz
import os
import numpy as np
import seaborn as sns

import torch
import torch.nn.functional as F

from peft import LoraConfig, get_peft_model

import shutil
from omegaconf import open_dict, OmegaConf

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from os.path import join as pjoin

# Function to load yaml configuration file
def load_config(root, config_path, config_name):
    """
    Function to load yaml configuration file
    
    Parameters
    ----------
    root : str
        Root directory
    config_path : str
        Path to the configuration file
    config_name : str
        Name of the configuration file
        
    Returns
    -------
    config : dict
        Dictionary containing the configuration
    """
    with open(pjoin(root, config_path, config_name)) as file:
        config = yaml.safe_load(file)
    return config

def fit_classical_ml_and_get_metrics(classifier, x_train, y_train, x_val, y_val):
    classifier.fit(x_train, y_train)
    train_accuracy = accuracy_score(y_train, classifier.predict(x_train))
    validation_accuracy = accuracy_score(y_val, classifier.predict(x_val))
    print('Accuracy Train: ', train_accuracy)
    print('Accuracy Test: ', validation_accuracy)
    return classifier, train_accuracy, validation_accuracy

def add_to_config(config, key, value):
    with open_dict(config):
        config[key] = value


def plot_loss_accuracy_curve(train_losses, 
                             val_losses,
                             train_accuracies,
                             val_accuracies,
                             save_path=''):
    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].plot(train_losses, label='train loss')
    ax[0].plot(val_losses, label='val loss')

    ax[1].plot(train_accuracies, label='train accuracy')
    ax[1].plot(val_accuracies, label='val accuracy')

    ax[0].grid()
    ax[1].grid()

    set_yticks(ax[0])
    set_yticks(ax[1])

    ax[0].legend()
    ax[1].legend()

    ax[0].set_xlabel('Epochs')
    ax[1].set_xlabel('Epochs')

    ax[0].set_ylabel('Loss')
    ax[1].set_ylabel('Accuracy')

    ax[0].set_title('Loss Curves')
    ax[1].set_title('Accuracy Curves')

    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'train_val_loss_accuracy.png'))
        plt.close()
    else:
        plt.show()

def plot_all_losses_and_accuracies_curve(train_losses, 
                                         val_losses, 
                                         train_lith_losses, 
                                         val_lith_losses,  
                                         train_phi_losses, 
                                         val_phi_losses, 
                                         train_sw_losses, 
                                         val_sw_losses,
                                         train_accuracies, 
                                         val_accuracies, 
                                         save_path=''):
    
    _, ax = plt.subplots(1, 5, figsize=(20, 5))

    ax[0].plot(train_losses, label='train')
    ax[0].plot(val_losses, label='val')
    ax[0].set_title('Total Loss')
    ax[0].legend()

    ax[1].plot(train_lith_losses, label='train')
    ax[1].plot(val_lith_losses, label='val')
    ax[1].set_title('Lith Loss')
    ax[1].legend()

    ax[2].plot(train_phi_losses, label='train')
    ax[2].plot(val_phi_losses, label='val')
    ax[2].set_title('Phi Loss')
    ax[2].legend()

    ax[3].plot(train_sw_losses, label='train')
    ax[3].plot(val_sw_losses, label='val')
    ax[3].set_title('Sw Loss')
    ax[3].legend()

    ax[4].plot(train_accuracies, label='train')
    ax[4].plot(val_accuracies, label='val')
    ax[4].set_title('Accuracy')
    ax[4].legend()

    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'train_val_all_losses_accuracy.png'))
        plt.close()
    else:
        plt.show()

def plot_true_vs_predicted_crossplot(gt, phi, sw, save_path=''):

    _, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].scatter(gt[:, :, 1].reshape(-1), phi[:, :].cpu().detach().numpy().reshape(-1))
    ax[0].plot([gt[:, :, 1].min().item(), gt[:, :, 1].max().item()], [gt[:, :, 1].min().item(), gt[:, :, 1].max().item()], color='red')
    ax[0].set_title('PHI')

    ax[1].scatter(gt[:, :, 2].reshape(-1), sw[:, :].cpu().detach().numpy().reshape(-1))
    ax[1].plot([gt[:, :, 2].min().item(), gt[:, :, 2].max().item()], [gt[:, :, 2].min().item(), gt[:, :, 2].max().item()], color='red')
    ax[1].set_title('SW')

    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'true_vs_predicted_crossplot.png'))
        plt.close()
    else:
        plt.show()

def plot_loss_curve(train_losses, val_losses,save_path=''):
    _, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(train_losses, label='train loss')
    ax.plot(val_losses, label='val loss')

    ax.grid()
    set_yticks(ax)
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.set_title('Loss Curves')
    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'train_val_loss.png'))
        plt.close()
    else:
        plt.show()

def set_yticks(ax, num_grid_lines=15):
    y_min, y_max = ax.get_ylim()
    step_size = (y_max - y_min) / (num_grid_lines - 1)
    y_grid_lines = [y_min + i * step_size for i in range(num_grid_lines)]
    ax.set_yticks(y_grid_lines)

def current_time():
    # Set the timezone to IST
    ist = pytz.timezone('Asia/Kolkata')

    # Get the current time in UTC
    utc_now = datetime.datetime.utcnow()

    # Convert UTC time to IST time
    ist_now = utc_now.replace(tzinfo=pytz.utc).astimezone(ist)

    # Format the IST time as a string
    ist_time_str = ist_now.strftime('%Y-%m-%d %HHr %MMin %SSec %Z%z')

    print("Current time in IST:", ist_time_str)

    return ist_time_str

def copy_running_code(target_folder):
    print('Copying running code...')
    current_directory = os.getcwd()
    ignore_content = ['data',
                    'outputs',
                    'results']
    files_in_current_directory = os.listdir(current_directory)

    shutil.rmtree(target_folder) if os.path.isdir(target_folder) else None

    # Create the target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Copy each file from the current directory to the target folder
    for file_name in files_in_current_directory:
        if not file_name.startswith('.') and not file_name in ignore_content:
            source_file_path = pjoin(current_directory, file_name)
            target_file_path = pjoin(target_folder, file_name)
            try:
                shutil.copytree(source_file_path, target_file_path)
            except:
                shutil.copyfile(source_file_path, target_file_path)

def load_model_from_checkpoint(model, checkpoint_name, exp_path, device):
    print('Loading model from checkpoint...')
    checkpoint = torch.load(pjoin(exp_path, checkpoint_name), map_location=device)
    checkpoint = checkpoint['model_state_dict']
    model.load_state_dict(checkpoint)
    return model, checkpoint

def plot_confusion_matrix(train, val, lithology_classes, save_path=''):
    _, ax = plt.subplots(1, 2, figsize=(15, 5))
    sns.heatmap(train, annot=True, 
                fmt='.2f', 
                xticklabels=lithology_classes.keys(), 
                yticklabels=lithology_classes.keys(), 
                ax = ax[0])
    sns.heatmap(val, annot=True, 
                fmt='.2f', 
                xticklabels=lithology_classes.keys(), 
                yticklabels=lithology_classes.keys(), 
                ax = ax[1])
    ax[0].set_title('Train')
    ax[1].set_title('Validation')
    ax[0].set_ylabel('True label')
    ax[0].set_xlabel('Predicted label')
    ax[1].set_ylabel('True label')
    ax[1].set_xlabel('Predicted label')
    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'confusion_matrix.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def save_best_checkpoint(model_chkpt, 
                         optim_chkpt, 
                         train_losses, 
                         val_losses, 
                         train_accuracies, 
                         val_accuracies, 
                         best_epoch, 
                         best_loss, 
                         trainer_config):
    
    # Save the model such that it can be used for resuming training from exact same point
    print("Saving the model...")
    model_info_dict = {
        'model_state_dict': model_chkpt,
        'optimizer_state_dict': optim_chkpt,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'best_epoch': best_epoch,
        'best_loss': best_loss
    }

    torch.save(
        model_info_dict, 
        pjoin(
            trainer_config['experiment_path'], 
            trainer_config['checkpoint_name']
        )
    )

def save_hparams(config, trainer_config):
    # Save the hyperparameters
    print("Saving the hyperparameters...")
    OmegaConf.save(
        config, 
        pjoin(
            trainer_config['experiment_path'], 
            trainer_config['hyperparameters_filename']
        )
    )

def update_best_metrices(
        val_loss, 
        val_accuracy, 
        epoch, 
        cm, 
        cm_val, 
        model, 
        optim, 
        best_loss, 
        best_accuracy, 
        best_epoch, 
        best_cm, 
        best_cm_val, 
        best_model_chkpt, 
        best_optim_chkpt
        ):

    if val_loss < best_loss:
        print(f"Model Performance Improved from epoch no. {best_epoch}")
        best_loss = val_loss
        best_accuracy = val_accuracy
        best_epoch = epoch
        best_cm_val = cm_val#.astype('float') / cm_val.sum(axis=1)[:, np.newaxis]
        best_cm = cm#.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        best_model_chkpt = model.state_dict()
        best_optim_chkpt = optim.state_dict()
    return best_loss, best_accuracy, best_epoch, best_cm, best_cm_val, best_model_chkpt, best_optim_chkpt

def load_policy_encoder_from_pretraining(model, encoder_checkpoint_path, device):
    print(f'Loading encoder from {encoder_checkpoint_path}')
    encoder_checkpoint = torch.load(encoder_checkpoint_path, map_location=device)['encoder_state_dict']
    model.vit_encoder.load_state_dict(encoder_checkpoint)
    print('Encoder loaded successfully...')
    return model

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

def load_lora(model, config, checkpoint_name, device):
    print_trainable_parameters(model)
    print('Loading model from already trained model')
    model, _ = load_model_from_checkpoint(model, checkpoint_name, config['trainer']['policy_exp_path'], device)

    print('Loading LoRA Config')
    lora_config = LoraConfig(**config['model']['lora'])
    print('Building LoRA model')
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model

def heatmap_precision_recall_f1_score(cm, val_cm, lithology_classes, save_path=''):
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (np.diag(cm).sum() / cm.sum())*100

    recall_val = np.diag(val_cm) / np.sum(val_cm, axis = 1)
    precision_val = np.diag(val_cm) / np.sum(val_cm, axis = 0)
    f1_val = 2 * precision_val * recall_val / (precision_val + recall_val)
    accuracy_val = (np.diag(val_cm).sum() / val_cm.sum())*100

    _, ax = plt.subplots(1, 2, figsize=(15, 5))

    sns.heatmap([precision, recall, f1], 
                annot=True, 
                fmt='.2f', 
                xticklabels=lithology_classes.keys(), 
                yticklabels=['Precision', 'Recall', 'F1'],
                ax = ax[0])
    
    sns.heatmap([precision_val, recall_val, f1_val],
                annot=True,
                fmt='.2f', 
                xticklabels=lithology_classes.keys(),
                yticklabels=['Precision', 'Recall', 'F1'],
                ax = ax[1])
    
    ax[0].set_title('Training Accuracy: {:.2f}%'.format(accuracy))
    ax[1].set_title('Validation Accuracy: {:.2f}%'.format(accuracy_val))
    ax[0].set_ylabel('Metrics')
    ax[0].set_xlabel('Lithology Classes')
    ax[1].set_ylabel('Metrics')
    ax[1].set_xlabel('Lithology Classes')

    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'performance_metrices.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def precision_recall_f1_score(cm, lithology_classes, save_path=''):
    recall = np.diag(cm) / np.sum(cm, axis = 1)
    precision = np.diag(cm) / np.sum(cm, axis = 0)
    f1 = 2 * precision * recall / (precision + recall)
    accuracy = (np.diag(cm).sum() / cm.sum())*100

    sns.heatmap([precision, recall, f1], annot=True, xticklabels=lithology_classes.keys(), yticklabels=['Precision', 'Recall', 'F1'])
    plt.title('Accuracy: {:.2f}%'.format(accuracy))
    if save_path not in [None, '', False]:
        plt.savefig(pjoin(save_path, 'performance_metrices.png'), bbox_inches='tight')
        plt.close()
    else:
        plt.show()