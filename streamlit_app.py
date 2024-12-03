import streamlit as st
import cv2
import whisper
import ffmpeg
from pydub import AudioSegment
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

groq_api_key = os.getenv('groq_api_key')

# Initialize the LLM with the API key and model
llm = ChatGroq(groq_api_key=groq_api_key, model='Llama-3.1-70b-Versatile')

# Define the prompt template for title generation
title_prompt = """Generate a meaningful 5-word title that reflects the crucifixion of Jesus based on the following content: {content}"""
description_prompt = """Provide a detailed summary of the crucifixion of Jesus based on the following passage, including the inscription, the two men crucified with Jesus, and the significance of the event: {content}"""

def get_video_metadata(video_path):
    """
    Extracts metadata from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Could not open video.")
        return None

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / frame_rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return frame_rate, frame_count, duration, width, height



def extract_audio_from_video(video_path, output_audio_path="audio.wav"):
    """
    Extracts audio from a video file and saves it as a WAV file using pydub.
    """
    # Load the video as an AudioSegment
    audio = AudioSegment.from_file(video_path)
    
    # Export the audio to a WAV file
    audio.export(output_audio_path, format="wav")
    
    return output_audio_path


def split_audio(audio_path, chunk_length=300):
    """
    Splits audio into smaller chunks for faster transcription.
    """
    audio = AudioSegment.from_file(audio_path)
    chunks = [audio[i * 1000: (i + chunk_length) * 1000] for i in range(0, len(audio) // 1000, chunk_length)]
    os.makedirs("chunks", exist_ok=True)
    chunk_files = []
    for i, chunk in enumerate(chunks):
        chunk_path = f"chunks/chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_files.append(chunk_path)
    return chunk_files

def transcribe_audio_chunks(chunk_files, model):
    """
    Transcribes each audio chunk and combines them into a single transcript.
    """
    transcripts = []
    for chunk in chunk_files:
        result = model.transcribe(chunk)
        transcripts.append(result["text"])
    return " ".join(transcripts)



def generate_title_and_description(transcript):
    # Define the prompt template for title generation
    title_prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Generate a meaningful title based on the following content: {content}"
    )

    # Define the prompt template for description generation
    description_prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Provide a detailed summary of  based on the following passage: {content}"
    )

    # Create LLMChain objects using the model and the prompt templates
    title_chain = LLMChain(prompt=title_prompt_template, llm=llm)
    description_chain = LLMChain(prompt=description_prompt_template, llm=llm)

    # Generate title using the title chain
    title_result = title_chain.run({"content": transcript})

    # Generate description using the description chain
    description_result = description_chain.run({"content": transcript})

    return title_result, description_result




def main():
    st.title("Video Metadata and Title/Description Generator (DataScience AI/ML)")

    # File uploader
    uploaded_file = st.file_uploader("Upload a Video (max size 30MB)", type=["mp4", "avi", "mov"])

    if uploaded_file:
        # Check file size
        if uploaded_file.size > 30 * 1024 * 1024:
            st.error("File size exceeds the 30MB limit!")
        else:
            st.video(uploaded_file)

            # Step 1: Save uploaded video to a temporary file
            temp_video_path = "temp_video.mp4"
            with open(temp_video_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Step 2: Extract video metadata
            st.write("Extracting video metadata...")
            metadata = get_video_metadata(temp_video_path)
            if metadata:
                frame_rate, frame_count, duration, width, height = metadata
                st.write(f"Frame Rate: {frame_rate} FPS")
                st.write(f"Frame Count: {frame_count}")
                st.write(f"Duration: {duration:.2f} seconds")
                st.write(f"Resolution: {width}x{height}")

            # Step 3: Extract audio from video
            # st.write("Extracting audio from video...")
            audio_path = extract_audio_from_video(temp_video_path)

            # Step 4: Split audio into smaller chunks (optional, for long videos)
            # st.write("Splitting audio into chunks...")
            chunk_files = split_audio(audio_path, chunk_length=300)  # 5-minute chunks

            # Step 5: Load Whisper model
            print("Loading Whisper model...")
            model = whisper.load_model("tiny")  # Use CPU

            # Step 6: Transcribe each chunk and combine
            print("Transcribing audio...")
            transcript = transcribe_audio_chunks(chunk_files, model)

            # Step 7: Generate title and description
            print("Generating title and description...")
            title, description = generate_title_and_description(transcript)

            # Step 8: Output the results
            st.subheader("Generated Title")
            st.write(title)
            st.subheader("Generated Description")
            st.write(description)

if __name__ == "__main__":
    main()
