# Import necessary libraries
import streamlit as st
import transformers
# from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf

# Load pre-trained T5 model and tokenizer
model = transformers.SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
processor = transformers.SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = transformers.SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

## Streamlit app header
st.title("Text-to-Speech System")

# Input text box
input_text = st.text_input("Enter Your Text Here:")

# Button to generate audio
if st.button("Generate Audio"):
    if input_text:
        # Tokenize and generate audio using T5 model
        inputs = processor(text=input_text, return_tensors="pt")

        # load xvector containing speaker's voice characteristics from a dataset
        embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
        
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
                
        # Save the generated audio
        sf.write("speech.wav", speech.numpy(), samplerate=16000)

        # Display audio player
        st.audio("speech.wav")
    else:
        st.warning("Please enter text before generating audio.")

# Streamlit app footer
st.write("Built with ❤️")
