# PixelRNN Implementation: PixelCNN, Row LSTM, and Diagonal BiLSTM

## Project Overview

This project implements and compares three generative models for image data based on the **Pixel Recurrent Neural Networks (PixelRNN)** paper by van den Oord et al. (2016). The goal is to reproduce the three main architectures proposed in the paper: **PixelCNN**, **Row LSTM**, and **Diagonal BiLSTM**, and evaluate their performance on the CIFAR-10 dataset.

## What We Wanted to Achieve

### Primary Objectives:
1. **Implement three distinct autoregressive generative models** for image generation
2. **Compare their performance** using negative log-likelihood (NLL) and bits per dimension metrics
3. **Generate high-quality sample images** from each model
4. **Analyze the trade-offs** between computational efficiency and generation quality

### Key Research Questions:
- How do different autoregressive architectures perform on CIFAR-10?
- What are the computational trade-offs between PixelCNN and PixelRNN variants?
- Which model achieves the best bits per dimension score?

## What We Implemented

### 1. PixelCNN Model
**Architecture**: Uses masked convolutions to ensure autoregressive generation
- **Mask Type A**: Applied to the first layer, masks center pixel and future pixels
- **Mask Type B**: Applied to subsequent layers, masks only future pixels
- **Residual Blocks**: Multiple masked convolutional layers with residual connections
- **Advantage**: Fully parallelizable during training, fast inference

**Key Features**:
- 7×7 masked convolution in first layer
- 3×3 masked convolutions in residual blocks
- ReLU activations with 1×1 convolutions
- 256-way softmax output for each RGB channel

### 2. Row LSTM Model
**Architecture**: Processes images row by row using LSTM
- **Row-wise Processing**: Each row is treated as a sequence
- **Input-to-State**: k×1 convolution for parallel computation
- **State-to-State**: Recurrent computation along rows
- **Triangular Receptive Field**: Captures context above each pixel

**Key Features**:
- Unidirectional LSTM processing
- k×1 convolution kernels (k≥3)
- Translation invariance along rows
- Sequential processing for generation

### 3. Diagonal BiLSTM Model
**Architecture**: Processes images diagonally using bidirectional LSTM
- **Skewing Operation**: Offsets rows to enable diagonal processing
- **Bidirectional Processing**: Left-to-right and right-to-left directions
- **Global Receptive Field**: Captures entire available context
- **2×1 Convolution Kernels**: Minimal information processing per step

**Key Features**:
- Input skewing and unskewing operations
- Bidirectional LSTM computation
- Full dependency field coverage
- Highly nonlinear computation

## Technical Implementation Details

### Data Preprocessing
- **Dataset**: CIFAR-10 (32×32×3 RGB images)
- **Pixel Values**: Discrete values (0-255) for discrete distribution modeling
- **Normalization**: Converted to [0,1] range then back to discrete values
- **Batch Processing**: Configurable batch sizes with data augmentation

### Training Configuration
- **Loss Function**: Negative log-likelihood (cross-entropy)
- **Optimizer**: Adam with weight decay
- **Learning Rate**: 1e-3 with ReduceLROnPlateau
- **Early Stopping**: Patience of 10 epochs
- **Validation Split**: 10% of training data

### Evaluation Metrics
- **Negative Log-Likelihood (NLL)**: Primary metric for model comparison
- **Bits per Dimension**: NLL normalized by image dimensions
- **Generated Sample Quality**: Visual inspection and statistical analysis
- **Training Efficiency**: Time and memory usage comparison

## What We Checked and Evaluated

### 1. Model Performance Metrics
- **NLL Scores**: Lower is better, measures how well the model predicts pixel values
- **Bits per Dimension**: Standardized metric for comparing generative models
- **Training Convergence**: How quickly each model reaches optimal performance
- **Validation Performance**: Generalization capability on unseen data

### 2. Generated Sample Quality
- **Visual Inspection**: Generated images compared to real CIFAR-10 samples
- **Statistical Analysis**: Pixel value distributions and diversity metrics
- **Sample Diversity**: Variety in generated images
- **Artifact Detection**: Identification of common generation artifacts

### 3. Computational Efficiency
- **Training Time**: Time required to train each model
- **Memory Usage**: GPU/CPU memory consumption
- **Inference Speed**: Time to generate single images
- **Scalability**: Performance with different image sizes

### 4. Architecture Analysis
- **Receptive Field**: How much context each model can access
- **Parallelization**: Training and inference parallelization capabilities
- **Gradient Flow**: How well gradients propagate through the network
- **Residual Connections**: Impact on training stability and performance

## Expected Results and Findings

### Performance Expectations (Based on Paper)
1. **Diagonal BiLSTM**: Should achieve the best NLL scores due to global receptive field
2. **Row LSTM**: Moderate performance with triangular receptive field
3. **PixelCNN**: Good performance with bounded receptive field, fastest training

### Key Insights
- **Receptive Field vs. Efficiency**: Larger receptive fields generally improve quality but reduce efficiency
- **Parallelization Trade-offs**: PixelCNN is fully parallelizable but has limited receptive field
- **Discrete vs. Continuous**: Discrete pixel modeling often performs better than continuous distributions
- **Residual Connections**: Critical for training deep autoregressive models

## Project Structure

```
Q3/
├── main.py                 # Main training and evaluation script
├── requirements.txt        # Python dependencies
├── README.md              # Project documentation
├── explanation.md         # This comprehensive explanation
├── configs/
│   └── config.py          # Configuration parameters
├── src/
│   ├── data_loader.py     # CIFAR-10 data loading and preprocessing
│   ├── models.py          # Model implementations (PixelCNN, Row LSTM, Diagonal BiLSTM)
│   ├── trainer.py         # Training logic and utilities
│   └── evaluator.py       # Evaluation metrics and visualization
├── models/                # Saved model checkpoints
├── results/               # Evaluation results and metrics
├── plots/                 # Generated plots and visualizations
└── logs/                  # Training logs and TensorBoard files
```

## Usage Instructions

### Training All Models
```bash
python main.py --model all --mode train_eval --epochs 100 --batch_size 32
```

### Training Specific Model
```bash
python main.py --model pixelcnn --mode train_eval --epochs 50
```

### Evaluation Only
```bash
python main.py --model all --mode eval
```

## Key Contributions

1. **Complete Implementation**: All three architectures from the original paper
2. **Comprehensive Evaluation**: Multiple metrics and visualizations
3. **Modular Design**: Clean, reusable code structure
4. **Detailed Documentation**: Thorough explanation of implementation choices
5. **Comparative Analysis**: Side-by-side comparison of all models

## Challenges and Solutions

### Implementation Challenges
1. **Masked Convolutions**: Complex masking logic for autoregressive generation
2. **Diagonal Processing**: Skewing and unskewing operations for Diagonal BiLSTM
3. **Memory Management**: Large models requiring careful memory optimization
4. **Training Stability**: Ensuring stable training for deep autoregressive models

### Solutions Implemented
1. **Custom MaskedConv2D Layer**: Proper masking implementation
2. **Efficient Diagonal Operations**: Optimized skewing/unskewing functions
3. **Gradient Clipping**: Prevented gradient explosion
4. **Residual Connections**: Improved gradient flow and training stability

## Future Improvements

1. **Multi-Scale Architecture**: Implementation of hierarchical generation
2. **Conditional Generation**: Class-conditional image generation
3. **Advanced Sampling**: Temperature scaling and nucleus sampling
4. **Efficiency Optimization**: Model compression and quantization
5. **Additional Metrics**: Inception Score and FID evaluation

## Conclusion

This implementation provides a comprehensive comparison of the three main PixelRNN architectures, demonstrating the trade-offs between model complexity, computational efficiency, and generation quality. The results help understand the fundamental differences between convolutional and recurrent approaches to autoregressive image generation, contributing to the broader understanding of generative modeling techniques.

The project successfully reproduces the key findings from the original PixelRNN paper while providing a clean, well-documented implementation that can serve as a foundation for further research in autoregressive generative models.
