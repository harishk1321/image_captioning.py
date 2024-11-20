import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# --- Preprocess Image ---
def preprocess_image(image_path, target_size=(224, 224)):
    image = load_img(image_path, target_size=target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image

# --- Extract Features Using ResNet50 ---
resnet_model = ResNet50(include_top=False, weights='imagenet')

def extract_features(image):
    feature_map = resnet_model.predict(image)
    feature_vector = feature_map.flatten()
    feature_vector = feature_vector / np.linalg.norm(feature_vector)  # Normalize
    return feature_vector

# --- Captioning Model (LSTM) ---
def create_captioning_model(vocab_size, max_sequence_length, embedding_dim=256, input_dim=2048):
    model = Sequential()
    model.add(Dense(256, input_dim=input_dim, activation='relu'))  # Dense layer for image features
    model.add(Dropout(0.5))
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_sequence_length))  # Embedding layer for captions
    model.add(LSTM(256, return_sequences=True))
    model.add(LSTM(256))
    model.add(Dense(vocab_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# --- Tokenizer and Preprocessing ---
def tokenize_captions(captions, vocab_size=10000):
    tokenizer = Tokenizer(num_words=vocab_size)
    tokenizer.fit_on_texts(captions)
    sequences = tokenizer.texts_to_sequences(captions)
    padded_sequences = pad_sequences(sequences, padding='post')
    return tokenizer, padded_sequences

# --- Generate Caption ---
def generate_caption(image, model, tokenizer, max_sequence_length=34):
    features = extract_features(image)  # Extract features from the image
    seq = [tokenizer.word_index['startseq']]  # Start sequence token
    caption = 'startseq'

    for _ in range(max_sequence_length):
        # Pad sequence for the LSTM
        padded_seq = pad_sequences([seq], maxlen=max_sequence_length, padding='post')
        
        # Predict next word
        pred = model.predict([features, padded_seq])
        pred_word_idx = np.argmax(pred)
        
        # Get word from index
        pred_word = tokenizer.index_word[pred_word_idx]
        caption += ' ' + pred_word
        
        # Stop if end token is predicted
        if pred_word == 'endseq':
            break
        
        # Update the sequence with predicted word
        seq.append(pred_word_idx)
    
    return caption.replace('startseq', '').replace('endseq', '')

# Example usage
image_path = 'path_to_image.jpg'  # Replace with the path to your image
processed_image = preprocess_image(image_path)

# Example captions for training (use a real dataset for better training)
captions = ["a man is playing guitar", "a person playing the guitar", "a guitar being played by a man"]
tokenizer, padded_captions = tokenize_captions(captions)

# Assuming vocab_size and max_sequence_length from your data
vocab_size = 10000  # Example vocab size
max_sequence_length = 34  # Example max sequence length

captioning_model = create_captioning_model(vocab_size, max_sequence_length)

# Example: Train the model (use actual training data here)
# captioning_model.fit([image_features, padded_captions], to_categorical(labels, num_classes=vocab_size), epochs=10, batch_size=32)

# Generate caption for a new image
caption = generate_caption(processed_image, captioning_model, tokenizer)
print(f"Generated Caption: {caption}")
