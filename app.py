import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.densenet import preprocess_input

import pickle
from PIL import Image
import tensorflow as tf

""" Load model and resources
model_path = "models/model.keras"
tokenizer_path = "models/tokenizer.pkl"
feature_extractor_path = "models/feature_extractor.keras"
caption_model = load_model(model_path)
feature_extractor = load_model(feature_extractor_path)
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)
"""
caption_model = load_model("models/model.keras")
feature_extractor = tf.keras.applications.DenseNet201(
    weights='imagenet', include_top=False, pooling='avg'
)
with open("models/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

max_length = 34  # Replace with actual max length used during training

def extract_features(image):
    # Resize to match expected input shape of MobileNetV2 (224x224)
    img = image.resize((224, 224))
    img = np.array(img)
    if img.shape[-1] == 4:  # Handle images with alpha channel
        img = img[..., :3]

    # Preprocess for MobileNetV2
    img = preprocess_input(img)
    img = np.expand_dims(img, axis=0)
    #img = tf.keras.applications.inception_v3.preprocess_input(img)
    #img = np.expand_dims(img, axis=0)
    features = feature_extractor.predict(img)
    return features

"""def generate_caption(photo):
    in_text = 'start'
    for _ in range(max_length):
        #sequence = tokenizer.texts_to_sequences([in_text])[0]
        #sequence = pad_sequences([sequence], maxlen=max_length)
        #yhat = caption_model.predict([photo, sequence], verbose=0)
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')
        yhat = caption_model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None or word == 'end':
            break
        in_text += ' ' + word
    return in_text.replace('start', '').strip()
# new previous chatgpt
def generate_caption(image):
    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding='post', truncating='post')
        
        yhat = caption_model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat, None)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()
    final_caption = final_caption[1:-1]  # remove startseq and endseq
    return ' '.join(final_caption)
def caption_image(image):
    photo = extract_features(image)
    caption = generate_caption(photo)
    return caption
iface = gr.Interface(fn=caption_image, 
                     inputs=gr.Image(type="pil"), 
                     outputs="text",
                     title="AI Image Caption Generator",
                     description="Upload an image and generate a caption.")
if __name__ == "__main__":
    iface.launch()
"""

def generate_caption(image):
    photo = extract_features(image)
    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length, padding='post', truncating='post')
        yhat = caption_model.predict([photo, seq], verbose=0)
        yhat_index = np.argmax(yhat)
        word = tokenizer.index_word.get(yhat_index)
        if word is None or word == "endseq": break
        in_text += " " + word

    tokens = in_text.split()[1:]  # drop 'startseq'
    if tokens and tokens[-1] == "endseq": tokens = tokens[:-1]
    return " ".join(tokens) or "Could not generate caption."

demo = gr.Interface(
    fn=generate_caption,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Caption Generator"
)

if __name__ == "__main__":
    demo.launch()

