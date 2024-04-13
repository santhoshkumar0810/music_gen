from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import base64


@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    return model


def generate_music_tensors(description, duration: int):
    print("Description: ", description)
    print("Duration: ", duration)
    model = load_model()

    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions=[description],
        progress=True,
        return_tokens=True
    )


    return output[0]


def save_audio(samples: torch.Tensor, music_name:str, music_format:str):
    print("Samples (inside function): ", samples)
    sample_rate = 32000
    save_path = "audio_output/"
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]
    supported_formats = ["wav", "mp3"]  # Define supported formats

    if music_format not in supported_formats:
        st.error(f"Unsupported format: {music_format}. Please choose from {supported_formats}")
        return None  # Indicate failure

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"{music_name}.{music_format}")
        torchaudio.save(audio_path, audio, sample_rate)
        return audio_path


def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href


st.set_page_config(
    page_icon="musical_note",
    page_title="AI Music Composer",
    layout="wide"
)


def main():

    hide_default_format = """
           <style>
           #MainMenu {visibility: hidden; }
           footer {visibility: hidden;}
           </style>
           """
    st.markdown(hide_default_format, unsafe_allow_html=True)
    st.title("Text to Music Generator🎵")

    with st.expander("See explanation"):
        st.write("Music Generator app built using Meta's Audiocraft library. We are using Music Gen Small model.")

    text_area = st.text_area("Enter your description.......")
    time_slider = st.slider("Select time duration (In Seconds)", 0, 60, 1)
    music_name = st.text_area("Enter a name to you file")
    music_format = st.selectbox(
        'Select the music format',
        ('mp3', 'wav'))
    generate_button = st.button("Generate Music")  # This line creates the button

    if generate_button:
        st.subheader("Generated Music")
        music_tensors = generate_music_tensors(text_area, time_slider)
        save_music_file = save_audio(music_tensors,music_name,music_format)
        audio_filepath = save_music_file
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()
