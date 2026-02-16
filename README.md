Detecting Spam Emails Using Tensorflow in Python

Great 🔥 Let’s build a Spam Email Detection Model using TensorFlow in Python step-by-step.

Since you're using the spam_ham_dataset.csv, I’ll give you a clean, working pipeline you can run directly.

📌 Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping

📌 Step 2: Load Dataset
data = pd.read_csv(r"C:\Users\KIIT\Downloads\spam_ham_dataset.csv")
data.head()


Check columns:

data.columns


Usually columns are like:

label

text

If your labels are text (spam, ham), convert them:

data['label'] = data['label'].map({'ham':0, 'spam':1})

📌 Step 3: Text Preprocessing

Download stopwords (only once):

# nltk.download('stopwords')


Clean text:

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

📌 Step 4: Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['clean_text'])

sequences = tokenizer.texts_to_sequences(data['clean_text'])
padded = pad_sequences(sequences, maxlen=100)

📌 Step 5: Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    padded, data['label'], test_size=0.2, random_state=42
)

📌 Step 6: Build LSTM Model
model = Sequential()

model.add(Embedding(input_dim=5000, output_dim=64, input_length=100))
model.add(LSTM(64, return_sequences=False))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

📌 Step 7: Train Model
early_stop = EarlyStopping(monitor='val_loss', patience=3)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

📌 Step 8: Evaluate Model
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)


You should get around 97–99% accuracy depending on preprocessing.

📌 Step 9: Test Custom Email
sample = ["Congratulations! You have won $1000. Click here now!"]

sample_seq = tokenizer.texts_to_sequences(sample)
sample_pad = pad_sequences(sample_seq, maxlen=100)

prediction = model.predict(sample_pad)

if prediction[0][0] > 0.5:
    print("Spam Email 🚨")
else:
    print("Not Spam ✅")

🎯 What We Built

Text Cleaning

Tokenization

Padding

LSTM Deep Learning Model

Spam Prediction System
