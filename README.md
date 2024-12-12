# lithology_petrophysical_parm_est_ARViT
This repo contains the code for the paper "Efficient self-attention based joint optimization for lithology and petrophysical parameter estimation in the Athabasca Oil Sands"

# 🚧 Hold Back, We're Doing Some Cosmetic Changes! 🚧

We're making some improvements to the codebase to make it easier for readers to go through. We will publish the code by the end of **December 2024**. Please check back soon!

Meanwhile, if you'd like to learn more about the proposed work, you can check out the [published paper](https://www.sciencedirect.com/science/article/abs/pii/S0926985124002489)

Thanks for your patience! 🙌

# Highlights

- ARViT for predicting Lithology & estimating Petrophysical Parameters.
- ARViT uses Self-Attention and AutoRegression to capture temporal relationship.
- Multitask Learning jointly Optimizes across three different tasks.
- LoRA efficiently reduces trainable parameters hence cutting GPU consumption.
- Joint optimization with LoRA enhances ARViT's training efficiency.

# Abstract
Accurately identifying lithology and petrophysical parameters, such as porosity and water saturation, are essential in reservoir characterization. Manual interpretation of well-log data, the conventional approach, is not only labor-intensive but also susceptible to human errors. To address these challenges of lithology identification and petrophysical parameter estimation in the Athabasca Oil Sands area, this study introduces an AutoRegressive Vision Transformer (ARViT) model for lithology and petrophysical parameter prediction. The effectiveness of ARViT lies in its self-attention mechanism and its ability to handle data sequentially, allowing the model to capture important spatial dependencies within the well-log data. This mechanism enables the model to identify subtle spatial and temporal relationships among various geophysical measurements. The model is also interpretable and can serve as an assistive tool for geoscientists, enabling faster interpretation while reducing human bias. The interpretable nature of the model should assist geoscientists in conducting faster quality checks of the predictions, ensuring that errors are not propagated to subsequent stages. This study adopts a multitask learning approach, jointly optimizing the model's performance across multiple tasks simultaneously. To evaluate the effectiveness of the ARViT model, we conducted series of experiments and comparisions, testing it against traditional artificial neural networks (ANN), Long Short-Term Memory (LSTM), and Vision Transformer (ViT) models. To showcase the versatility of ARViT, we apply Low-Rank Adaptation (LoRA) to a different smaller dataset, showing its potential to adapt to different geological contexts. LoRA not only helps in model adaptability but also helps to reduce the number of trainable parameters. Our findings demonstrate that ARViT outperforms ANN, LSTM, and ViT in estimating lithological and petrophysical parameters. While lithology prediction has been a well-explored field, ARViT's unique blend of features, including its self-attention mechanism, autoregression, and multitask approach along with efficient fine tuning using LoRA, sets it apart as a valuable tool for the complex task of lithology prediction and petrophysical parameter estimation.


## Publication Link
https://www.sciencedirect.com/science/article/abs/pii/S0926985124002489
