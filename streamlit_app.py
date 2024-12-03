from flask import Flask, request, jsonify, render_template
import cv2
import whisper
from pydub import AudioSegment
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

app = Flask(__name__)

groq_api_key = os.getenv('groq_api_key')

# Initialize the LLM with the API key and model
llm = ChatGroq(groq_api_key=groq_api_key, model='Llama-3.1-70b-Versatile')


def get_video_metadata(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / frame_rate
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()
    return frame_rate, frame_count, duration, width, height


def extract_audio_from_video(video_path, output_audio_path="audio.wav"):
    audio = AudioSegment.from_file(video_path)
    audio.export(output_audio_path, format="wav")
    return output_audio_path


def split_audio(audio_path, chunk_length=300):
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
    transcripts = []
    for chunk in chunk_files:
        result = model.transcribe(chunk)
        transcripts.append(result["text"])
    return " ".join(transcripts)


def generate_title_and_description(transcript):
    title_prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Generate a meaningful title based on the following content: {content}"
    )

    description_prompt_template = PromptTemplate(
        input_variables=["content"],
        template="Provide a detailed summary based on the following passage: {content}"
    )

    title_chain = LLMChain(prompt=title_prompt_template, llm=llm)
    description_chain = LLMChain(prompt=description_prompt_template, llm=llm)

    title_result = title_chain.run({"content": transcript})
    description_result = description_chain.run({"content": transcript})

    return title_result, description_result


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if a file was uploaded
        if "video" not in request.files:
            return jsonify({"error": "No video file provided"}), 400

        video_file = request.files["video"]

        if video_file.filename == "":
            return jsonify({"error": "No selected file"}), 400

        # Save the file to a temporary location
        video_path = os.path.join("uploads", video_file.filename)
        os.makedirs("uploads", exist_ok=True)
        video_file.save(video_path)

        # Simulate metadata generation for simplicity
        metadata = {
            "frame_rate": "30 FPS",
            "frame_count": "4500",
            "duration": "150 seconds",
            "resolution": "1920x1080"
        }

        # Simulate title and description generation
        title = "Generated Title"
        description = "Generated description of the video."

        # Return metadata, title, and description as JSON
        return jsonify({
            "metadata": metadata,
            "title": title,
            "description": description
        })

    return """
    <!doctype html>
    <title>Video Upload</title>
    <h1>Upload Video</h1>
    <form action="/" method="POST" enctype="multipart/form-data">
        <input type="file" name="video">
        <input type="submit" value="Upload">
    </form>
    """

if __name__ == "__main__":
    app.run(debug=True)
