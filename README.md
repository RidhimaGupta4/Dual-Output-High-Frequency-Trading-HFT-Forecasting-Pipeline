# Dual-Output Forecasting of Mid-Price Movements in High-Frequency LOB Data

## 📌 Project Overview
This project develops a high-frequency trading (HFT) machine learning pipeline designed to extract real-time insights from granular Limit Order Book (LOB) data. Using Amazon stock data (s092215-50-AMZN), the system employs a dual-output approach to simultaneously predict:
* **Regression**: Mid-price levels.
* **Classification**: 3-class directional price movements (Down, Stationary, Up).

## 🛠️ Technical Architecture
The framework integrates several deep learning architectures to capture market micro-dynamics:
* **Recurrent Neural Networks**: LSTM layers (256 units) to model temporal dependencies and historical context.
* **Convolutional Neural Networks**: 1D-Conv layers (64 filters) for detecting local patterns in short-term LOB dynamics.
* **Multi-Task Learning**: A Dual-Task LSTM with a shared temporal encoder to jointly optimize regression and classification.
* **MLP Variants**: Shallow, Deep, and Very Deep networks serving as high-dimensional baselines.
* **Ensemble Modeling**: Majority voting for classification and regression averaging to mitigate individual model bias.

## 🧪 Feature Engineering & Optimization
* **Enriched Feature Set**: Raw LOB data augmented with Lag features, Rolling Means, and Exponential Moving Averages (EMA).
* **Imbalance Handling**: Addressed the dominant stationary class using **Weighted Focal Loss**, **ADASYN** synthetic sampling, and stratified batch generation.
* **Validation Strategy**: Employed 5-fold **TimeSeriesSplit** to prevent information leakage and ensure the model only trains on past data.

## 📊 Key Performance Results

| Model | Avg. RMSE (Regression) | Accuracy (Classification) | Macro F1-Score |
| :--- | :--- | :--- | :--- |
| **Ensemble** | **169.84** | 0.4565 | 0.30 |
| **CNN** | 305.40 | **0.8961** | 0.31 |
| **LSTM** | 238.14 | 0.8813 | **0.39** |
| **Dual-Task** | 860.59 | 0.8880 | 0.33 |

* **Ensemble Regression**: Achieved the lowest error by mitigating individual model variance.
* **LSTM Performance**: Provided the best balance for minority class capture (directional movements).

## 🚀 Trading Strategy Integration
The model maps predicted classes to actionable HFT signals:
* **Class 0**: Short/Sell signal.
* **Class 1**: Hold/Neutral.
* **Class 2**: Long/Buy signal.
