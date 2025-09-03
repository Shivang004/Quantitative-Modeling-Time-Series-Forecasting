# 📈 Stock Price Prediction using LSTMs

This project implements a **Long Short-Term Memory (LSTM)** neural network using **TensorFlow** and **Keras** to predict future stock prices based on historical data.

The core of this project is a **two-model architecture**:
- A **stateless model** is used for efficient training.  
- Its learned weights are then transferred to a **stateful model** designed for accurate, multi-step **autoregressive** prediction.

---

## 📑 Table of Contents
- [How It Works](#how-it-works)
- [Project Features](#project-features)
- [Setup and Installation](#setup-and-installation)
  - [Prerequisites](#prerequisites)
  - [Dependencies](#dependencies)
- [Usage](#usage)
- [Configuration](#configuration)
- [Code Overview](#code-overview)

---

## ⚙️ How It Works

Predicting a time series **one step at a time** requires the model to **remember the immediate past**. This is called **statefulness**.  
However, training is most efficient when performed on **large, independent batches of data**, which is **stateless**.

This project resolves that conflict with a **two-model approach**:

### 🔹 Training Model (`model`)
- **Stateless** (`stateful=False`)
- Processes large batches of sequences: `(batch_size, num_unrollings, 1)`
- Learns general patterns in stock price data
- **Memory is reset** after each batch

### 🔹 Prediction Model (`prediction_model`)
- **Stateful** (`stateful=True`)
- Identical architecture to training model, but `batch_size=1`
- **Imports the trained weights**
- Predicts **one step at a time** and retains its state
- Supports **autoregressive forecasting**, where the model’s output becomes the next input

✅ This method combines **efficient training** with **accurate sequential prediction**.

---

## 🚀 Project Features
- **Custom Data Generator**: `DataGeneratorSeq` efficiently creates batches of sequential data  
- **Stacked LSTM Architecture**: Multiple LSTM layers capture complex temporal patterns  
- **Stateless Training**: Enables faster training with shuffled batches  
- **Stateful Prediction**: Maintains internal state for coherent multi-step forecasting  
- **Model Warm-up**: Prepares the prediction model with recent historical data  
- **Autoregressive Forecasting**: Uses its own predictions as input for future predictions  
- **Evaluation**: Calculates **Mean Squared Error (MSE)** for accuracy assessment  

---

## 🛠️ Setup and Installation

### Prerequisites
- **Python 3.7+**
- Pre-processed **NumPy array** of stock prices  
  (e.g., from the `Close` or `Adj Close` column of a CSV file)

### Dependencies
Install required libraries:
```bash
pip install tensorflow numpy pandas matplotlib
```

---

## 📌 Usage

### 1) Prepare Your Data
Prepare a **NumPy array** named `all_mid_data`.  

**Example:**
```python
# import pandas as pd
# df = pd.read_csv('your_stock_data.csv')
# all_mid_data = df['Close'].values
```

---

### 2) Run the Script
Run from terminal:
```bash
python your_script_name.py
```

---

### 3) Output
- Trains the **stateless LSTM model**
- Creates the **stateful prediction model** and copies weights
- Generates **multi-step predictions** from test data
- Outputs **average Test MSE** across sequences

---

## ⚡ Configuration

Adjustable parameters in the script:

- **`D`** → Input dimensionality *(default: 1 for univariate prediction)*  
- **`num_unrollings`** → Length of input sequence  
- **`batch_size`** → Sequences per training batch  
- **`num_nodes`** → Units in each LSTM layer *(e.g., `[200, 200, 150]`)*  
- **`dropout`** → Dropout rate for regularization  
- **`epochs`** → Training iterations over dataset  

---

## 🧩 Code Overview

- **`DataGeneratorSeq`** → Creates sequential input/label batches for training  
- **`create_lstm_model()`** → Defines stateless LSTM training model  
- **`prepare_training_data()`** → Builds training dataset (`X_train`, `y_train`)  
- **`create_prediction_model()`** → Defines stateful model for predictions  
- **Training Block** → Compiles and trains stateless model  
- **Prediction Block** →  
  - Copies weights to prediction model  
  - Iterates through test points (`test_points_seq`)  
  - Resets state, *warms up* with recent history  
  - Generates autoregressive predictions  

---

## 📊 Example Output (Illustrative)

```
Training loss: 0.0018
Validation loss: 0.0021
Average Test MSE: 0.0024
```

*(Your results will vary depending on dataset and parameters)*

---

## 📜 License
This project is released under the **MIT License**. You are free to use, modify, and distribute it with attribution.

---
