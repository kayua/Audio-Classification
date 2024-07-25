# Sound classification - Neural Network Models

Algorithms for mosquito flapping audio classification based on deep neural networks.



![Spectrogramas](Layout/AudioSegments.png?raw=true "Examples of traces: ground truth (obtained with 27 monitors), failed (obtained with 7 monitors/20 failed), and recovered (using NN).")
[View original publication](https://www.sciencedirect.com/science/article/pii/S1746809424004002)
## Neural Network Topologies

Six different deep neural network topologies are implemented and evaluated for the problem of identifying mosquitoes from the audio emitted by their wings.
### Original Papers:

1. Audio Spectrogram Transformer [https://arxiv.org/abs/2104.01778]
2. Long Short Term Memory [https://www.bioinf.jku.at/publications/older/2604.pdf]
3. Conformer [https://arxiv.org/abs/2005.08100]
4. Wav2Vec2 [https://arxiv.org/abs/2006.11477]
5. Residual [https://doi.org/10.1016/j.bspc.2024.106342]
6. MLP [https://ieeexplore.ieee.org/document/8942209]

<table>
    <tbody>
        <tr>
            <th width="20%">AST Topology</th>
            <th width="20%">LSTM Topology</th>
            <th width="20%">Conformer Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/AST-Model.png"></td>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/LSTM-Model.png"></td>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/Conformer-Model.png"></td>
        </tr>
    <tbody>
        <tr>
            <th width="20%">Wav2Vec2 Topology</th>
            <th width="20%">Residual Topology</th>
            <th width="20%">MLP Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/Wav2Vec2-Model.png"></td>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/Residual-Model.png"></td>
            <td><img src="https://github.com/kayua/ModelsAudioClassification/blob/main/Layout/MultiLayerPerceptron-Model.png"></td>
        </tr>

</table>

## Experimental Evaluation

### Fitting Analysis
Impact of the number of epochs on average error for Dense topology (arrangements A=3, window width W=11), LSTM topology (arrangements A=3, window width W=11), and Conv. topology (arrangements A=8, squared window width W=H=256).

<table>
    <tbody> 
        <tr>
            <th width="10%">AST Topology</th>
            <th width="10%">LSTM Topology</th>
            <th width="10%">Conformer Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/dense_error.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/lstm_error.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/conv_error.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
        </tr>
   <tbody> 
        <tr>
            <th width="10%">Wav2Vec2</th>
            <th width="10%">Residual</th>
            <th width="10%">MLP</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/dense_error.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/lstm_error.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/conv_error.png" alt="2018-06-04 4 43 02" style="max-width:100%;"></td>
        </tr>

</table>




###  Parameter Sensitivity Analysis

Parameter sensitivity of Conv. topology withuniform probabilistic injected failure Fprob =10%
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional Topology</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/sens_conv.png" alt="2018-06-04 4 33 16" style="max-width:50%;"></td>
        </tr>


</table>


### Comparing our Neural Networks
Comparison of topologies MLP, LSTM (LS), and CNN for probabilistic injected failure and monitoring injected failure.
<table>
    <tbody> 
        <tr>
            <th width="10%">Probabilistic Injected Failure</th>
            <th width="10%">Monitoring Injected Failure</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_pif.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/comparison_nn_mif.png" alt="2018-06-04 4 40 06" style="max-width:101%;"></td>
        </tr>
        
</table>

### Comparison with the State-of-the-Art (Convolutional vs Probabilistic)

Comparison between the best neural network model and state-of-the-art probabilistic technique. Values obtained for probabilistic error injection and monitoring error injection.
<table>
    <tbody>
        <tr>
            <th width="20%">Convolutional vs Probabilistic</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/results.png" alt="2018-06-04 4 33 16" style="max-width:120%;"></td>
        </tr>


</table>

### Qualitative Analysis

Impact, in terms of number (left) and duration (right) of a trace (S1) failed (Fmon = 20) and regenerated using the proposed BB-based (topology=Conv., threshold α =0.50, arrangements A =8, squared window width W = H =256) and prior probabilistic-based (threshold α =0.75).

<table>
    <tbody> 
        <tr>
            <th width="10%">Sessions Duration</th>
            <th width="10%">Number Sessions</th>
        </tr>
        <tr>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_duration.png" alt="2018-06-04 4 33 16" style="max-width:100%;"></td>
            <td><img src="https://github.com/kayua/Regenerating-Datasets-With-Convolutional-Network/blob/master/layout/CDF_number_sessions.png" alt="2018-06-04 4 40 06" style="max-width:100%;"></td>
        </tr>
        
</table>

## Steps to Install:

1. Upgrade and update
    - sudo apt-get update
    - sudo apt-get upgrade 
    
2. Installation of application and internal dependencies
    - git clone [https://github.com/kayua/ModelsAudioClassification]
    - pip install -r requirements.txt
    
3. Test installation:
    - python3 main.py -h


## Run experiments:

###  Run (all F_prob experiments)
`python3 run_jnsm_mif.py -c lstm`

### Run (only one F_prob scenario)
`python3 main.py`

###  Run (all F_mon experiments)
`python3 run_mif.py -c lstm`

### Run (only one F_mon scenario)
`python3 main_mif.py`


### Input parameters:

    Arguments(run_TNSM.py):
        
       -h, --help            Show this help message and exit

    --------------------------------------------------------------
   
    Arguments(main.py):

          -h, --help            Show this help message and exit

        --------------------------------------------------------------




## Requirements:

`matplotlib 3.4.1`
`tensorflow 2.4.1`
`tqdm 4.60.0`
`numpy 1.18.5`

`keras 2.4.3`
`setuptools 45.2.0`
`h5py 2.10.0`




