import streamlit as st
import librosa
from panns_inference import AudioTagging, SoundEventDetection, labels
from pydub import AudioSegment
import matplotlib.pyplot as plt

st.write("A Simple app to find what you're looking for in your very own audio files!")
results_labels = []
intensity_dict = {}
   
uploaded_file = st.file_uploader("Upload File", type=['mp3'])
if uploaded_file is not None:
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
    sound = AudioSegment.from_mp3(uploaded_file)
    dst = "test.wav"
    uploaded_file_wav = sound.export(dst, format="wav")

with st.form(key='significance_form'):
	significance_value = st.slider(label='Select Intensity', min_value=0.05, max_value=1.0, value=0.2, step=0.05, key="intensity")
	find_significance_button = st.form_submit_button(label="Find significant classes in video")
    
if find_significance_button and uploaded_file is not None:
    (audio, _) = librosa.core.load(uploaded_file_wav, sr=32000, mono=True)
    audio = audio[None, :]

    at = AudioTagging(checkpoint_path=None, device='cuda')
    (clipwise_output, embedding) = at.inference(audio)
    audio_tagging_values = []
    results_index = []
    
    for i in range(len(labels)):
        if(clipwise_output[0][i] > significance_value):
            audio_tagging_values.append(clipwise_output[0][i])
            results_index.append(i)
    results_labels = [labels[i] for i in results_index]
    
    for index in results_index:
        intensity_dict[index]= []
    
    sed = SoundEventDetection(checkpoint_path=None, device='cuda')
    framewise_output = sed.inference(audio)
    
    for frame in framewise_output[0]:
        for i in range(len(frame)):
            for key, value in intensity_dict.items():
                if(i == key):
                    value.append(frame[i])
    
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Intensity")
    for key, value in intensity_dict.items():
        ax.plot(range(len(value)), value, label=labels[key])
    ax.legend()
    st.pyplot(plt)

with st.form(key='label_form'):
    selected_options = st.multiselect("If the graph has too many classes you can add the ones you want to see here and refresh the graph", 
                             results_labels, default = results_labels, key="options")
    st.write('You selected:', selected_options)
    submitted = st.form_submit_button(label="Show graph only with chosen classes")

if submitted:
    fig = plt.figure(figsize=(30,15))
    ax = fig.add_subplot(1,1,1)
    ax.set_xlabel("Frame number")
    ax.set_ylabel("Intensity")
    st.write(selected_options)
    for key, value in intensity_dict.items():
        st.write(labels[key])
        if(labels[key] in selected_options):
            ax.plot(range(len(value)), value, label=labels[key])
    ax.legend()
    st.pyplot(plt)

    
    
    

        
    
