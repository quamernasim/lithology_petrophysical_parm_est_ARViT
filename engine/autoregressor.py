import numpy as np
from tqdm import tqdm

import torch
from einops import rearrange
import torch.nn as nn

from sklearn.metrics import confusion_matrix

from utils.misc import update_best_metrices

def train_engine(
        epoch, 
        model, 
        train_loader, 
        regression_criterion,
        classification_criterion, 
        optimizer, 
        num_epochs, 
        loss_weights,
        device,
        patch_size,
        num_classes
    ):

    train_loss = 0.0
    sw_train_loss = 0.0
    lith_train_loss = 0.0
    phi_train_loss = 0.0
    train_correct = 0

    gt, pred = [], []

    model.train()
    for batch_inputs, batch_labels in tqdm(train_loader, 
                                           total=len(train_loader), 
                                           desc=f"Train - Epoch {epoch+1}/{num_epochs}"):
        optimizer.zero_grad()

        batch_inputs = batch_inputs.to(device)
        batch_labels = batch_labels.to(device).to(torch.float32)

        # autoregressive outputs initialized with zeros
        outputs = torch.zeros((batch_inputs.shape[0], patch_size, num_classes+2)).to(device)
        vit_embedding = None

        # Forward pass
        for step in range(patch_size):
            lith_next_token, phi_next_token, sw_next_token, vit_embedding = model(batch_inputs, outputs, step, vit_embedding)
            next_token = torch.cat([lith_next_token, phi_next_token, sw_next_token], axis = -1)
            updated_initilizer = outputs.clone()
            updated_initilizer[:, step, :] = next_token
            outputs = updated_initilizer

        lith_batch_labels = batch_labels[:, :, 0]
        phi_output_labels = batch_labels[:, :, 1:2]
        sw_output_labels = batch_labels[:, :, 2:]

        lith_output = outputs[:, :, :num_classes]
        phi_output = outputs[:, :, num_classes:num_classes+1]
        sw_output = outputs[:, :, num_classes+1:]

        lith_batch_labels = lith_batch_labels.long()

        outputs_ = rearrange(lith_output, 'b n d -> b d n')
        lith_loss = classification_criterion(outputs_, lith_batch_labels)

        phi_loss = regression_criterion(phi_output, phi_output_labels)

        # sw_batch_labels_gradient = torch.gradient(sw_output_labels, dim = 1)[0]
        # sw_output_gradient = torch.gradient(sw_output, dim = 1)[0]
        # sw_gradient_loss = regression_criterion(sw_output_gradient, sw_batch_labels_gradient)
        # sw_std_loss_val = regression_criterion(sw_output.squeeze().std(dim=1), sw_output_labels.squeeze().std(dim=1))
        # sw_mean_loss_val = regression_criterion(sw_output_val.squeeze().mean(dim=1), sw_batch_labels_val.squeeze().mean(dim=1))
        sw_loss = regression_criterion(sw_output, sw_output_labels)

        loss = (
            (lith_loss*loss_weights[0]) + 
            (phi_loss*loss_weights[1]) + 
            (sw_loss*loss_weights[2])
            )

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        predicted = torch.argmax(nn.Softmax(dim = -1)(lith_output), dim=-1)
        train_correct += (((lith_batch_labels == predicted).sum(-1).float().mean().item())/batch_inputs.shape[1])*100

        gt.append(lith_batch_labels.cpu())
        pred.append(predicted.cpu())

        train_loss += loss.item()
        sw_train_loss += sw_loss.item()
        phi_train_loss += phi_loss.item()
        lith_train_loss += lith_loss.item()

    cm = confusion_matrix(torch.cat(gt, dim=0).view(-1), torch.cat(pred, dim=0).view(-1))

    # Calculate average training loss and accuracy for the epoch
    train_loss /= len(train_loader)
    sw_train_loss /= len(train_loader)
    lith_train_loss /= len(train_loader)
    phi_train_loss /= len(train_loader)
    train_accuracy = train_correct / len(train_loader)

    return train_loss, lith_train_loss, phi_train_loss, sw_train_loss, train_accuracy, cm

def validation_engine(
        epoch, 
        model, 
        val_loader, 
        regression_criterion,
        classification_criterion,
        num_epochs, 
        loss_weights,
        device,
        patch_size,
        num_classes
    ):
    
    val_loss = 0.0
    sw_val_loss = 0.0
    lith_val_loss = 0.0
    phi_val_loss = 0.0
    val_correct = 0

    gt_val, pred_val = [], []
    
    # Evaluate on the validation set
    model.eval()
    for batch_inputs_val, batch_labels_val in tqdm(val_loader, 
                                                   total=len(val_loader), 
                                                   desc=f"Val - Epoch {epoch+1}/{num_epochs}"):
        
        batch_inputs_val = batch_inputs_val.to(device)
        batch_labels_val = batch_labels_val.to(device).to(torch.float32)
        # autoregressive outputs initialized with zeros
        val_outputs = torch.zeros((batch_inputs_val.shape[0], patch_size, num_classes+2)).to(device)
        val_vit_embedding = None

        with torch.no_grad():
            for step in range(patch_size):
                lith_output_val, phi_output_val, sw_output_val, val_vit_embedding = model(batch_inputs_val, val_outputs, step, val_vit_embedding)
                val_next_token = torch.cat([lith_output_val, phi_output_val, sw_output_val], axis = -1)
                val_updated_initilizer = val_outputs.clone()
                val_updated_initilizer[:, step, :] = val_next_token
                val_outputs = val_updated_initilizer

        lith_batch_labels_val = batch_labels_val[:, :, 0]
        phi_batch_labels_val = batch_labels_val[:, :, 1:2]
        sw_batch_labels_val = batch_labels_val[:, :, 2:]

        lith_output = val_outputs[:, :, :num_classes]
        phi_output = val_outputs[:, :, num_classes:num_classes+1]
        sw_output = val_outputs[:, :, num_classes+1:]

        lith_batch_labels_val = lith_batch_labels_val.long()

        outputs_ = rearrange(lith_output, 'b n d -> b d n')
        lith_loss_val = classification_criterion(outputs_, lith_batch_labels_val)

        phi_loss_val = regression_criterion(phi_output, phi_batch_labels_val)

        # sw_batch_labels_gradient_val = torch.gradient(sw_batch_labels_val, dim = 1)[0]
        # sw_output_gradient_val = torch.gradient(sw_output, dim = 1)[0]
        # sw_gradient_loss_val = regression_criterion(sw_output_gradient_val, sw_batch_labels_gradient_val)
        # sw_std_loss_val = regression_criterion(sw_output.squeeze().std(dim=1), sw_batch_labels_val.squeeze().std(dim=1))
        # sw_mean_loss_val = regression_criterion(sw_output_val.squeeze().mean(dim=1), sw_batch_labels_val.squeeze().mean(dim=1))
        sw_loss_val = regression_criterion(sw_output, sw_batch_labels_val)

        loss_val = (
            (lith_loss_val*loss_weights[0]) +
            (phi_loss_val*loss_weights[1]) +
            (sw_loss_val*loss_weights[2])
            )
        
        # Calculate validation accuracy
        val_predicted = torch.argmax(nn.Softmax(dim = -1)(lith_output), dim=-1)
        val_correct += (((val_predicted == lith_batch_labels_val).sum(-1).float().mean().item())/batch_inputs_val.shape[1])*100

        val_loss += loss_val.item()
        sw_val_loss += sw_loss_val.item()
        lith_val_loss += lith_loss_val.item()
        phi_val_loss += phi_loss_val.item()

        gt_val.append(lith_batch_labels_val.cpu())
        pred_val.append(val_predicted.cpu())

    cm_val = confusion_matrix(torch.cat(gt_val, dim=0).view(-1), torch.cat(pred_val, dim=0).view(-1))

    val_loss /= len(val_loader)
    sw_val_loss /= len(val_loader)
    lith_val_loss /= len(val_loader)
    phi_val_loss /= len(val_loader)
    val_accuracy = val_correct / len(val_loader)

    return val_loss, lith_val_loss, phi_val_loss, sw_val_loss, val_accuracy, cm_val

def train(
        num_epochs, 
        model, 
        train_loader, 
        val_loader, 
        regression_criterion,
        classification_criterion,
        loss_weights,
        optimizer, 
        tolerance,
        device,
        patch_size,
        num_classes
    ):
    
    train_losses = []
    val_losses = []

    train_lith_losses = []
    val_lith_losses = []

    train_phi_losses = []
    val_phi_losses = []

    train_sw_losses = []
    val_sw_losses = []

    train_accuracies = []
    val_accuracies = []

    best_loss = np.inf
    best_cm_val = None
    best_cm = None
    best_model_chkpt = None
    best_optim_chkpt = None
    best_epoch = 1
    best_accuracy = 0.0

    for epoch in range(num_epochs):
        (
            train_loss, 
            lith_train_loss, 
            phi_train_loss, 
            sw_train_loss, 
            train_accuracy, 
            cm
        ) = train_engine(epoch, 
                         model, 
                         train_loader, 
                         regression_criterion,
                         classification_criterion,
                         optimizer, 
                         num_epochs,
                         loss_weights,
                         device,
                         patch_size,
                         num_classes
        )
        
        (
            val_loss, 
            lith_val_loss, 
            phi_val_loss, 
            sw_val_loss, 
            val_accuracy, 
            cm_val
        ) = validation_engine(epoch, 
                              model, 
                              val_loader, 
                              regression_criterion, 
                              classification_criterion,
                              num_epochs,
                              loss_weights,
                              device,
                              patch_size,
                              num_classes
        )

        train_losses.append(train_loss)
        train_lith_losses.append(lith_train_loss)
        train_phi_losses.append(phi_train_loss)
        train_sw_losses.append(sw_train_loss)
        train_accuracies.append(train_accuracy)

        val_losses.append(val_loss)
        val_lith_losses.append(lith_val_loss)
        val_phi_losses.append(phi_val_loss)
        val_sw_losses.append(sw_val_loss)
        val_accuracies.append(val_accuracy)

        # Print the progress for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Train Lith Loss: {lith_train_loss:.4f}, Train Phi Loss: {phi_train_loss:.4f}, " \
                f"Train Sw Loss: {sw_train_loss:.4f}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, " \
                    f"Val Lith Loss: {lith_val_loss:.4f}, Val Phi Loss: {phi_val_loss:.4f}, Val Sw Loss: {sw_val_loss:.4f}, " \
                        f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        (
            best_loss, 
            best_accuracy, 
            best_epoch, 
            best_cm, 
            best_cm_val, 
            best_model_chkpt, 
            best_optim_chkpt
        ) = update_best_metrices(
            val_loss, 
            val_accuracy, 
            epoch, 
            cm, 
            cm_val, 
            model, 
            optimizer, 
            best_loss, 
            best_accuracy, 
            best_epoch, 
            best_cm, 
            best_cm_val, 
            best_model_chkpt, 
            best_optim_chkpt
        )

        if epoch - best_epoch > tolerance:
            print("Early stopping")
            break

    return (train_losses, 
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
            best_model_chkpt, 
            best_optim_chkpt)