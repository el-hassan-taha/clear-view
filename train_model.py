import pandas as pd
import numpy as np
import re
import os
import pickle
import json
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SimpleRNN

# Download NLTK data
nltk.download('stopwords')

def clean_reuters_bias(text):
    # Remove "(Reuters) - " or similar at the beginning
    return re.sub(r'^.*?\(Reuters\)\s*-\s*', '', text)

def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return ' '.join(words)

def main():
    print("Loading data...")
    # Check if dataset.csv already exists to speed up or re-create it
    if os.path.exists('data/dataset.csv'):
        df = pd.read_csv('data/dataset.csv')
    else:
        true_df = pd.read_csv('data/true.csv')
        fake_df = pd.read_csv('data/fake.csv')

        true_df['label'] = 1
        fake_df['label'] = 0

        print("Cleaning Reuters bias...")
        true_df['text'] = true_df['text'].apply(clean_reuters_bias)

        print("Concatenating and shuffling...")
        df = pd.concat([true_df, fake_df], ignore_index=True)
        df = df[['title', 'text', 'label']]
        df = shuffle(df, random_state=42).reset_index(drop=True)
        
        print("Saving merged dataset...")
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/dataset.csv', index=False)
        print("Dataset saved to data/dataset.csv")

    if 'clean_text' not in df.columns:
        print("Preprocessing text...")
        df['clean_text'] = df['text'].apply(preprocess_text)
        df.to_csv('data/dataset.csv', index=False)

    print("Splitting Data...")
    df['clean_text'] = df['clean_text'].fillna('')
    X_text = df['clean_text'].astype(str).to_numpy()
    y = df['label'].to_numpy()
    X_train_text, X_test_text, y_train, y_test = train_test_split(X_text, y, test_size=0.2, random_state=42)
    
    os.makedirs('models', exist_ok=True)
    metadata = {}

    # ---------------------------
    # Tokenization and Padding
    # ---------------------------
    print("Tokenizing and Padding...")
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(X_train_text)
    
    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=100)
    X_test_pad = pad_sequences(X_test_seq, maxlen=100)

    # ---------------------------
    # Model A: RNN
    # ---------------------------
    print("--- Training RNN ---")
    print("Building RNN model...")
    rnn_model = Sequential()
    rnn_model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    rnn_model.add(SimpleRNN(64))
    rnn_model.add(Dense(1, activation='sigmoid'))

    rnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training RNN...")
    rnn_model.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

    print("Evaluating RNN...")
    rnn_loss, rnn_acc = rnn_model.evaluate(X_test_pad, y_test)
    print(f"RNN Accuracy: {rnn_acc:.4f}")
    
    metadata["rnn"] = round(rnn_acc, 4)

    print("Saving RNN assets...")
    rnn_model.save('models/rnn_model.h5')
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(tokenizer, f)

    # ---------------------------
    # Model B: LSTM
    # ---------------------------
    print("--- Training LSTM ---")
    print("Building LSTM model...")
    model = Sequential()
    model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
    model.add(LSTM(64))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Training LSTM...")
    model.fit(X_train_pad, y_train, epochs=5, batch_size=128, validation_data=(X_test_pad, y_test))

    print("Evaluating LSTM...")
    # Evaluate on test set
    lstm_loss, lstm_acc = model.evaluate(X_test_pad, y_test)
    print(f"LSTM Accuracy: {lstm_acc:.4f}")
    
    metadata["lstm"] = round(lstm_acc, 4)

    print("Saving LSTM assets...")
    model.save('models/lstm_model.h5')

    # ---------------------------
    # Save Metadata
    # ---------------------------
    print("Saving metadata.json...")
    with open('models/metadata.json', 'w') as f:
        json.dump(metadata, f)

    print("All models trained and saved successfully!")

if __name__ == "__main__":
    main()
