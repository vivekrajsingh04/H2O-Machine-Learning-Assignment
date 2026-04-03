# Data Analytics: Introduction to Machine Learning using H2O

## Introduction

This report details the methodology, results, and conclusion of an introductory machine learning pipeline built using the H2O framework. The objective of this assignment is to understand the basic and advanced functionalities provided by H2O, particularly focusing on data preprocessing, traditional machine learning models (Random Forest), Neural Networks, Deep Learning, and Mathematical Optimization.

## Methodology

### 1. Dataset Selection & Preprocessing
The **Iris dataset** from the UCI Machine Learning Repository was selected for this classification task. It contains four features (sepal length, sepal width, petal length, and petal width) and a target variable (class/species) with three potential labels.
- **Handling Missing Values**: The dataset has no missing values natively, but to demonstrate generalized preprocessing capabilities in H2O, numerical columns were imputed with median values (`h2o.impute()`).
- **Data Transformation**: The target variable `class` was explicitly transformed into a categorical factor using `.asfactor()` to ensure H2O treats this as a multi-class categorization task rather than regression.
- **Data Splitting**: The dataset was split into training (80%) and testing (20%) datasets.

### 2. Machine Learning Approaches
Three distinct modeling approaches were adopted to evaluate the performance:
1. **Basic Machine Learning (Random Forest)**: As an introductory baseline, an `H2ORandomForestEstimator` was trained with 50 trees and a maximum depth of 5.
2. **Simple Neural Network**: A basic neural architecture via `H2ODeepLearningEstimator` consisting of a single hidden layer of size 10 and executing over 20 epochs with a `Tanh` activation function.
3. **Optimized Deep Learning**: A hyperparameter tuning grid search (`H2OGridSearch`) was incorporated, searching various `hidden` topologies `([20, 20], [50, 50])`, `epochs`, and regularization `L1` penalties spanning across architectures with `RectifierWithDropout`.

### 3. Mathematical Optimization and Loss Function
When training classification models, particularly Neural Networks, a mathematical loss function is minimized through mathematical optimization (such as Stochastic Gradient Descent).
- **Optimization Strategy**: For Deep Learning models within H2O, optimization iterations iteratively attempt to minimize objective loss using ADADELTA (an adaptive learning rate method). It finds local minima on the cost curve iteratively by computing gradients with respect to weights.
- **Loss Function Identification**: For this multi-class problem, **Cross-Entropy Loss (Logloss)** is employed. It evaluates the probabilities output by the classifier. The optimal model holds a Logloss closer to 0.

## Results

*The following metrics were derived after running the models over the 20% test partition.*

**1. Random Forest Classifier**
- **Logloss**: 0.1009
- **Mean Per Class Error**: ~0.033 (based on Logloss performance mapping)

**2. Simple Neural Network Classifier**
- **Logloss**: 0.1839

**3. Tuned Deep Learning Network Classifier**
- **Configuration Selected**: Hidden Layers: [50, 50], Epochs: 50.0
- **Logloss**: 0.0915

### Performance Comparison
| Model | Test Logloss |
|-------|--------------|
| Random Forest | 0.1009 |
| Simple NN | 0.1839 |
| Tuned DL | 0.0915 |

## Conclusion

By comparing the resulting Logloss values, the Tuned Deep Learning network configured with [50, 50] hidden layers across 50 epochs emerged as the most optimal architecture with a minimal logloss of 0.0915. It outperformed the simple single-layer NN (0.1839 logloss) and slightly edged out the traditional Random Forest model (0.1009 logloss).

*We observe how H2O empowers straightforward orchestration of robust classical algorithms and extensible deep learning structures in parallel compute-ready environments.*
