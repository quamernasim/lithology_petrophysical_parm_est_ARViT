# lithology_petrophysical_parm_est_ARViT
This Repo contains code for paper Efficient Self-Attention Based Joint Optimization for Lithology and Petrophysical Parameter Estimation in the Athabasca Oil Sands

## Main Contributions
The main contributions of this paper are as follows:
1. We propose a novel approach for lithology and petrophysical parameter estimation in the Athabasca Oil Sands area using the AutoRegressive Vision Transformer (ARViT) model.
2. We demonstrate the effectiveness of ARViT in estimating lithology and petrophysical parameters, such as core-calibrated porosity and water saturation, in the Athabasca Oil Sands area.
3. We show that ARViT outperforms traditional artificial neural networks (ANN), Long Short-Term Memory (LSTM), and Vision Transformer (ViT) models in estimating lithology and petrophysical parameters.
4. We demonstrate that ARViT is a versatile model that can be adapted to different geological contexts using Low-Rank Adaptation (LoRA).

## Data Description
The data used in this study is from the Athabasca Oil Sands area in Alberta, Canada. The data is provided by the Alberta Energy Regulator (AER) and is publicly available. The data is available in the following link: https://ags.aer.ca/publication/spe-006

## How to run the code
- The code is written in Python 3.9.13. PyTorch is used to develop the model.

- As opposed to argparser we used hydra to manage the hyperparameters. The hyperparameters are stored in the config folder. Hence no need to have argparser in the code. This makes the code more readable and easy to use.

### To run the code:
- Clone the repository
- Install the requirements using the following command:
```bash
pip install -r requirements.txt
```
- To run the code, use the following command:
```bash
python train.py # This will run the code with default parameters
```
- To run the code with different parameters, use the following command:
```bash
python train.py trainer.optim='adam' trainer.lr=0.001 model.mlp_dim=512 callbacks.early_stopping_tolerance=10 data.x_file_name='X.h5' # This will run the code with different parameters. You can change the parameters as per your need based on the config files.
```

## Results
The results of the model trained on the Athabasca Oil Sands area are present in the results folder. It has got result from the following models:
1. ANN - ```results/dl.ann```
2. ViT - ```results/dl.vit```
3. ARViT - ```results/dl.vit```
4. ARViT + LoRA - ```results/dl.lora```
5. LSTM - ```notebooks/ml/prediction_using_rnn.ipynb```

## Inference
The inference code is present in the notebooks folder. The notebooks are written in Python 3.9.13. The notebooks are as follows:
1. ```notebooks/ml/policy_inference_ann.ipynb``` - This notebook contains the code for inference using ANN.
2. ```notebooks/ml/prediction_using_rnn.ipynb``` - This notebook contains the code for training and inference using LSTM.
3. ```notebooks/ml/policy_inference_vit.ipynb``` - This notebook contains the code for inference using ViT.
4. ```notebooks/dl/policy_inference_arvit.ipynb``` - This notebook contains the code for inference using ARViT.
5. ```notebooks/dl/policy_inference_tl.ipynb``` - This notebook contains the code for inference using ARViT + Transfer Learning on LoRA dataset.
6. ```notebooks/dl/policy_inference_scratch.ipynb``` - This notebook contains the code for inference using ARViT from scratch on LoRA dataset.
7. ```notebooks/dl/policy_inference_lora.ipynb``` - This notebook contains the code for inference using ARViT + LoRA.


