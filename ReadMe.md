# Sound classification - Convolutional Neural Networks 

This repository contains the implemented code for the classification of mosquito audios using deep neural networks. It includes
the algorithm Residual model, the analysis and  categorization of complex acoustic patterns. 

The original paper: Residual (https://doi.org/10.1016/j.bspc.2024.106342) 

## Topology of model 
![Residual !](/Layout/Residual-Model.png "Residual-Model")
#### 
| Training Parameters  description | Evaluated Value                                        |
|----------------------------------|:-------------------------------------------------------|
| Epochs                           | Total number of training epochs **[10, 20, 30]**       |
| Learning Rate                    | Learning rate used **[0.1, 0.01, 0.001]**              |
| Loss Function                    | Loss function employed **[Categorical Cross-Entropy]** |
| Optimization Algorithm           | Optimization algorithm used **[Adam]**                 |
| Number of Folds                  | Number of folds for cross-validation **[10]**          |
| Batch Size                       | Batch size for training **[32]**                       |
| Sample Rate                      | Sample rate of sounds **[8000]**                       |
| Segment Length                   | Length of sound segment **[40, 60]**                   | 
---
### Training Curve
![Residual !](/Results/ResidualModel_loss.png "ResidualModel_loss")

---
## Evaluation Analysis 
### Confusion Matrices
![Residual !](/Results/matrix_6.png "matrix_6")

---
### ROC Curve 
![Residual !](/Results/ROC_ResidualModel.png "ROC_ResidualModel")

---
| Arguments                           |                                               |
|:------------------------------------|:----------------------------------------------|
| residual_hop_length                 | Hop length for STFT                           |
| residual_window_size_factor         | Factor applied to FFT window size             |
| residual_number_filters_spectrogram | Number of filters for spectrogram generation  |
| residual_filters_per_block          | Number of filters in each convolutional block |
| residual_file_extension             | File extension for audio files                |
| residual_dropout_rate               | Dropout rate in the network                   |
| residual_number_layers              | Number of convolutional layers                |
| residual_optimizer_function         | Optimizer function to use                     |
| residual_overlap                    | Overlap between patches in the spectrogram    |
| residual_loss_function              | Loss function to use during training          |
| residual_decibel_scale_factor       | Scale factor for converting to decibels       |
| residual_convolutional_padding      | Padding type for convolutional layers         |
| residual_input_dimension            | input dimension of the model                  |
| residual_intermediary_activation    | Activation function for intermediary layers   |
| residual_last_layer_activation      | Activation function for the last layer        |
| residual_size_pooling               | Size of the pooling layers                    |
| residual_window_size                | Size of the FFT window                        |
| residual_size_convolutional_filters | Size of the convolutional filters             | 




















