# PubMed Abstract Segmentation Using Hybrid Embeddings with CNN, Bi-LSTM, and Positional Encoding

## Overview
This project implements a deep learning model to classify sentences in PubMed RCT abstracts into five sections: **BACKGROUND**, **OBJECTIVES**, **METHODS**, **RESULTS**, and **CONCLUSIONS**. The model explores the capabilities of **Natural Language Processing (NLP)** using sequential sentence classification to segment abstracts and extract structured information.

This project replicates the methodologies from the papers **PubMed 200k RCT** and **Neural Networks for Joint Sentence Classification in Medical Abstracts**, leveraging hybrid embeddings (token-level, character-level, and positional encodings) for robust sentence classification.


## Key Technologies and Concepts

- **Natural Language Processing (NLP)** for text segmentation and classification.
- **Hybrid Embedding**s: Combination of token-level embeddings (pre-trained from TensorFlow Hub), character-level embeddings, and positional encodings (line number and total lines in abstract).
- **Neural Networks**: Utilizes **Convolutional Neural Networks (CNN)**, **Bidirectional LSTMs (Bi-LSTM)**, and **Fully Connected Layers** for classification.
- **TensorFlow** and **Keras** for model building and training.
- **Positional Encoding** to enhance sequential learning by integrating metadata such as sentence position within the abstract.

## Project Structure
The project consists of the following components:
```
.
├── pubmed-rct/                           # Directory containing PubMed 200k RCT dataset
│   ├── PubMed_200k_RCT/                  # Full dataset
│   ├── PubMed_20k_RCT_numbers_replaced/  # Subset of dataset with numbers anonymized
├── trained_models/                       # Directory for storing trained models
│   ├── cnn_lstm_token_char_model.h5      # Trained CNN+Bi-LSTM hybrid model
├── src/                                  # Source code
│   ├── data_preprocessing.py             # Data extraction, cleaning, and preprocessing scripts
│   ├── model_training.py                 # Main script for model architecture and training
│   ├── embedding_layers.py               # Implementation of token, character, and positional embeddings
│   ├── model_evaluation.py               # Scripts for model evaluation and results analysis
├── README.md                             # This file
├── requirements.txt                      # Python dependencies
```

## Dataset
**Source**: The dataset comes from PubMed 200k RCT, consisting of around 200,000 Randomized Controlled Trial (RCT) abstracts.
**Labels**: Sentences in abstracts are labeled into five categories: BACKGROUND, OBJECTIVES, METHODS, RESULTS, and CONCLUSIONS.
**File structure**:
- `train.txt`: Training data.
- `dev.txt`: Validation data.
- `test.txt`: Test data.

## Prerequisites
Ensure you have Python 3.x installed, along with the following dependencies:
```
pip install tensorflow keras numpy pandas matplotlib scikit-learn tensorflow-hub
```

## Dataset
The dataset is stored in password_Data.sqlite and contains two key columns:
- password: The actual password string.
- strength: The strength of the password, where 0 = Weak, 1 = Normal, and 2 = Strong.

## Steps and Methodology
1. **Data Processing**

**Data Extraction**: Load and process the dataset using pandas. The abstracts are segmented into sentences, and each sentence is labeled with one of the five categories.
**Cleaning**: Preprocess the data by converting text to lowercase, removing special characters, and handling missing values.

2. **Embedding Layers**

**Token Embeddings**: Pre-trained embeddings from TensorFlow Hub (Universal Sentence Encoder) are used to capture the semantic meaning of the words in each sentence.
**Character Embeddings**: A character-level embedding layer is built to capture sub-word information using a **Convolutional Neural Network (CNN)**.
**Positional Encoding**: **One-hot encodings** of sentence position (line number) and total lines in the abstract are used to capture positional information, making the model aware of sentence ordering.

3. **Model Architecture**

**Token-Level Model**: Uses pre-trained embeddings to encode the text and passes them through a **Dense Layer**.
**Character-Level Model**: Character embeddings are passed through a **Bidirectional LSTM (Bi-LSTM)** to capture sequential patterns at the character level.
**Combined Hybrid Model**: Token and character embeddings are combined using **Concatenation**. The model also incorporates positional encodings (line number and total lines) to enhance learning.
**Classifier**: The combined embeddings are passed through a **CNN**, followed by fully connected layers, ending with a **softmax** output to classify sentences into one of five categories

4. **Model Training**

The model is trained using **Categorical Crossentropy Loss** and the **Adam Optimizer**.
**Label smoothing** is applied to reduce overfitting by softening overly confident predictions.
The model is trained for several epochs with **Dropout** layers for regularizatio

5. **Evaluation and Results**

The model is evaluated on the validation and test datasets.
Metrics such as **accuracy**, **precision**, **recall**, and **F1-score** are calculated using the **scikit-learn** library.
The final model achieves ~83% accuracy on the validation set.

6. **Saving the Model**

The trained model, along with the vectorizers and embedding layers, is saved in `.h5` format for future inference.


## Example Code Snippets
### Token and Character Embeddings:

```
# Token Embedding (Pre-trained from TensorFlow Hub)
token_embedding = hub.KerasLayer("https://tfhub.dev/google/universal-sentence-encoder/4", trainable=False)

# Character Embedding (Trainable CNN)
char_embedding = layers.Embedding(input_dim=NUM_CHAR_TOKENS, output_dim=25, mask_zero=False)
```
### Combining Embeddings and Training:
```
# Combine token and character embeddings
combined_embeddings = layers.Concatenate()([token_output, char_lstm_output])

# Train the model with Dropout and Dense layers
x = layers.Dense(256, activation="relu")(combined_embeddings)
x = layers.Dropout(0.5)(x)
output = layers.Dense(5, activation="softmax")(x)

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.fit(train_dataset, epochs=10, validation_data=val_dataset)
```

## Conclusion
This project demonstrates the application of advanced NLP techniques for segmenting scientific abstracts. By leveraging hybrid embeddings (token, character, and positional), we achieve accurate sentence classification, contributing to the structured extraction of information from medical literature.

