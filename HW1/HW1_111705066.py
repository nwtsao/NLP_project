import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Flatten, Concatenate, Dropout, LSTM, Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import json

vocab_size = 5000
embedding_dim = 50
max_length_title = 10
max_length_text = 50
dense_units = 50
dropout_rate = 0.5
learning_rate = 0.0004

# call back epoch
class EpochOutputCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(f"第 {epoch+1} 個 epoch 結束:")
        print(f"  訓練損失: {logs['loss']:.4f}")
        print(f"  訓練準確率: {logs['accuracy']:.4f}")
        if 'val_loss' in logs and 'val_accuracy' in logs:
            print(f"  驗證損失: {logs['val_loss']:.4f}")
            print(f"  驗證準確率: {logs['val_accuracy']:.4f}")
        print("------------------------")

epoch_output_callback = EpochOutputCallback()

with open('train.json', 'r', encoding='utf-8') as file:
    train_data = json.load(file)

title_data = [item['title'] for item in train_data]
text_data = [item['text'] for item in train_data]
verified_purchase_data = [int(item['verified_purchase']) for item in train_data]
helpful_vote_data = [item['helpful_vote'] for item in train_data]
rating_data = [item['rating'] for item in train_data]

# tokenization and padding
tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(title_data + text_data)
title_sequences = tokenizer.texts_to_sequences(title_data)
text_sequences = tokenizer.texts_to_sequences(text_data)
title_padded = pad_sequences(title_sequences, maxlen=max_length_title, padding='post')
text_padded = pad_sequences(text_sequences, maxlen=max_length_text, padding='post')

# adjust rating
adjusted_rating_data = [rating - 1 for rating in rating_data]

# Bi-LSTM
title_input = Input(shape=(max_length_title,), name='title')
text_input = Input(shape=(max_length_text,), name='text')
verified_purchase_input = Input(shape=(1,), name='verified_purchase')
helpful_vote_input = Input(shape=(1,), name='helpful_vote')

embedding = Embedding(vocab_size, embedding_dim)
title_embedded = embedding(title_input)
text_embedded = embedding(text_input)

title_bilstm = Bidirectional(LSTM(32))(title_embedded)
text_bilstm = Bidirectional(LSTM(32))(text_embedded)

dense_verified = Dense(16, activation='relu')(verified_purchase_input)
dense_helpful = Dense(16, activation='relu')(helpful_vote_input)

merged = Concatenate()([title_bilstm, text_bilstm, dense_verified, dense_helpful])
dense = Dense(50, activation='relu')(merged)
dense = Dropout(dropout_rate)(dense)
output = Dense(5, activation='softmax', name='rating')(dense)

bilstm_model = Model(inputs=[title_input, text_input, verified_purchase_input, helpful_vote_input], outputs=output)

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
bilstm_model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

early_stopping_callback = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
# train model
bilstm_model.fit(
    {
        'title': title_padded,
        'text': text_padded,
        'verified_purchase': np.array(verified_purchase_data).reshape(-1, 1),
        'helpful_vote': np.array(helpful_vote_data).reshape(-1, 1)
    },
    np.array(adjusted_rating_data),
    batch_size=256,
    epochs=25,
    validation_split=0.4,
    callbacks=[epoch_output_callback, early_stopping_callback]
)

# test
with open('test.json', 'r', encoding='utf-8') as file:
    test_data = json.load(file)

test_title_data = [item['title'] for item in test_data]
test_text_data = [item['text'] for item in test_data]
test_verified_purchase_data = [int(item['verified_purchase']) for item in test_data]
test_helpful_vote_data = [item['helpful_vote'] for item in test_data]

test_title_sequences = tokenizer.texts_to_sequences(test_title_data)
test_text_sequences = tokenizer.texts_to_sequences(test_text_data)
test_title_padded = pad_sequences(test_title_sequences, maxlen=max_length_title, padding='post')
test_text_padded = pad_sequences(test_text_sequences, maxlen=max_length_text, padding='post')

predictions = bilstm_model.predict({
    'title': test_title_padded,
    'text': test_text_padded,
    'verified_purchase': np.array(test_verified_purchase_data).reshape(-1, 1),
    'helpful_vote': np.array(test_helpful_vote_data).reshape(-1, 1)
})

predicted_ratings = np.argmax(predictions, axis=1) + 1

# output
output_data = {'index': [f'index_{i}' for i in range(len(predicted_ratings))], 'rating': predicted_ratings}
output_df = pd.DataFrame(output_data)
output_df.to_csv('test_lstm3.csv', index=False)

print("預測結果已輸出到 test_lstm3.csv")
