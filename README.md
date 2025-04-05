# Audio Deepfake Detection Project

## Overview

I've implemented an audio deepfake detection system based on the AASIST model, which uses graph attention networks to identify AI-generated speech. This project includes model research, implementation, and analysis using the ASVspoof 2019 dataset.

## Model Research & Selection

After digging through several approaches in the [Audio-Deepfake-Detection](https://github.com/media-sec-lab/Audio-Deepfake-Detection) repo, I narrowed down to these three promising models:

### 1. AASIST: Audio Anti-Spoofing with Graph Attention Networks

* **What makes it cool**: Uses a graph structure to connect spectral and temporal features, letting the model find patterns across both domains simultaneously.
* **Performance**: 0.83% EER on ASVspoof 2019 LA dataset – pretty impressive!
* **Why I liked it**: Processes raw audio directly without needing separate feature extraction steps. The graph approach seems particularly good at finding subtle fakeness patterns.
* **Downsides**: Somewhat complex architecture might be tricky to deploy efficiently.

### 2. RawNet2 with Sinc Filters

* **Main innovation**: Processes raw waveforms directly with learnable filters and uses GRUs for temporal modeling.
* **Performance**: 1.12% EER on ASVspoof 2019 LA.
* **Strengths**: Simpler architecture than AASIST but still effective. Good balance of accuracy vs. speed.
* **Limitations**: Not quite as accurate as AASIST on benchmark tests.

### 3. ResNet with Light CNN Features

* **Technical approach**: Combines ResNet's deep learning with LCNN's efficient feature extraction.
* **Results**: 2.50% EER (LA dataset) and 0.46% EER (PA dataset).
* **Advantages**: Uses well-established architectures, potentially more interpretable.
* **Drawbacks**: Slightly higher computational needs and requires separate feature extraction.

**Why I chose AASIST**: After weighing the options, I went with AASIST because it had the best performance metrics and I was intrigued by the graph-based approach. I thought its end-to-end architecture would be a good fit for real-time analysis of conversations.

## Implementation Process

Here's how I built my implementation:

### Key Components

* **Raw waveform processing** using SincConv layers to learn frequency-specific features
* **Residual blocks** for robust feature extraction
* **Graph attention network** to model relationships between audio features
* **Memory-efficient data handling** with custom caching mechanisms

### Biggest Challenges & My Solutions

1. **Dataset path nightmare**
   * The ASVspoof dataset has a confusing directory structure that kept breaking my code
   * I built a robust path detection system with multiple fallbacks to find files regardless of exact structure

2. **Kaggle resource limitations**
   * RAM capacity constraints made loading the full dataset challenging
   * Limited GPU utilization (only using 0.1-0.2% of available memory) despite having a P100 GPU
   * Created a custom LRU cache that automatically purges least-used files when memory gets tight

3. **Slow training despite available resources**
   * Each epoch took about 6 minutes for training and 1.5 minutes for validation
   * Implemented Automatic Mixed Precision (AMP) training and optimized batch processing
   * Still struggled to utilize GPU resources effectively (only 24MB out of 16GB used)

4. **Generalization gap**
   * Model achieved near-perfect 0.0018 EER on validation but 0.1562 on evaluation
   * Created comprehensive audio augmentation with noise addition, time shifting, and speed changes
   * Added mixup data augmentation during training to reduce overfitting

## Results & Analysis

Here's how my implementation performed through training:

* Started with an EER of 0.4368 (Epoch 1)
* Improved to 0.3115 (Epoch 2) and 0.2497 (Epoch 3)
* By Epoch 10: 0.0047 EER with AUC reaching 0.9999
* Best validation result: 0.0018 EER at Epoch 19 with perfect AUC of 1.0000

Final evaluation on the test set:
* Accuracy: 0.6281
* AUC: 0.9285
* EER: 0.1562
* min-tDCF: 0.5000

There's a gap between validation and test performance (0.0018 vs 0.1562 EER), suggesting some overfitting to the validation set. This is a common challenge in deepfake detection where models can learn dataset-specific patterns rather than generalizable features.

### What Worked Well

* The model learned relevant features directly from raw audio without manual feature engineering
* Memory optimization through caching kept training stable despite Kaggle's RAM constraints
* Achieved impressive validation metrics (0.0018 EER, perfect AUC) within 20 epochs
* Graph-based approach seemed effective at capturing complex audio artifacts

### What Could Be Better

* Massive underutilization of GPU resources - only using ~24MB of 16GB available on P100
* Training took ~6 minutes per epoch despite having a powerful GPU
* Significant gap between validation and test performance indicating generalization issues
* Complex architecture with many hyperparameters made optimization difficult in limited time

## Reflections & Future Improvements

### Personal Takeaways

The most challenging aspects of this project were:

* **Resource management on Kaggle**: Fighting with Kaggle's limitations was frustrating. I spent hours trying to optimize GPU utilization but still only used 0.2% of available memory. My P100 GPU was basically sitting idle while training crawled along.
* **Architecture implementation**: Getting the graph attention mechanism working correctly with batch processing took several attempts. The mathematical complexity of GAT made implementation tricky.
* **Bridging the validation-test gap**: The stark difference between validation performance (0.0018 EER) and test results (0.1562 EER) highlights how difficult it is to create models that truly generalize to new deepfake types.

### What I'd Improve Next

If I had more time, I'd focus on:

1. **Model optimizations**:
   * Increase model capacity to match the original paper's implementation 
   * Try transformer components for better temporal modeling
   * Experiment with knowledge distillation for a lighter, faster model

2. **Training enhancements**:
   * Implement adaptive batch sizes based on available GPU memory
   * Add cyclical learning rates for better convergence
   * Gather more diverse training data to improve generalization

3. **Production-ready features**:
   * Quantize the model to reduce inference time
   * Optimize for streaming audio processing
   * Create smaller specialized models for different device targets

### Deployment Thoughts

To make this production-ready, I'd:

* Optimize the model through quantization and pruning
* Set up a microservice architecture with proper scaling
* Implement comprehensive monitoring and feedback collection
* Establish a continuous improvement pipeline with automated retraining
* Design appropriate confidence thresholds based on the specific use case

Overall, this project gave me valuable insights into audio deepfake detection and the challenges of implementing advanced deep learning architectures for real-world applications.
