import gradio as gr
import cv2
import whisper
import spacy
from PIL import Image
from diffusers import StableDiffusionPipeline #stable model to be updated if used
import torch
import logging
import os
import io

# Disable WANDB logging and configure logging level
logging.disable(logging.WARNING)
os.environ["WANDB_DISABLED"] = "true"

# Load models
whisper_model = whisper.load_model("base")
spacy.prefer_gpu()
spacy_nlp = spacy.load("en_core_web_sm")

#Initialize the model
stable_diffusion_pipeline = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-2",
    torch_dtype=torch.float16
).to("cuda" if torch.cuda.is_available() else "cpu")


# # Set up the DALL·E 3 XL V2 API
# # Set your API key directly (or via environment variable if you prefer)
# api_key = "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"

# print("API Key:", api_key)

# # Define the Hugging Face API URL for the DALL·E 3 XL V2 model
# API_URL = "https://api-inference.huggingface.co/models/ehristoforu/dalle-3-xl-v2"
# headers = {"Authorization": f"Bearer {api_key}"}

print("All models loaded successfully!")

# def query(payload):
#     try:
#         response = requests.post(API_URL, headers=headers, json=payload)
#         if response.status_code == 200:
#             return response.content
#         else:
#             logging.error(f"Error in API request: {response.status_code} - {response.text}")
#             return None
#     except Exception as e:
#         logging.error("Exception during API request:", exc_info=e)
#         return None
def extract_keyframes(video_path, frame_interval=30, num_frames=5):
    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        success, frame = cap.read()
        count = 0
        while success and count < num_frames:
            if count % frame_interval == 0:
                frames.append(frame)
            success, frame = cap.read()
            count += 1
        cap.release()
        return frames
    except Exception as e:
        logging.error("Error extracting keyframes:", exc_info=e)
        return None
def test_extract_keyframes():
    video_path = "video.mp4"
    frames = extract_keyframes(video_path)

    assert frames is not None, "Keyframe extraction failed"
    assert len(frames) > 0, "No keyframes extracted"
    print("Keyframe extraction test passed")

test_extract_keyframes()
def transcribe_audio(video_path):
    try:
        result = whisper_model.transcribe(video_path)
        return result['text']
    except Exception as e:
        logging.error("Error transcribing audio:", exc_info=e)
        return None
def test_transcribe_audio():
    video_path = "video.mp4"
    transcription = transcribe_audio(video_path)

    assert transcription is not None, "Transcription failed"
    assert len(transcription) > 0, "Empty transcription"
    print("Transcription test passed")

test_transcribe_audio()
def extract_keywords(text):
    try:
        if not text or not text.strip():
            logging.warning("Empty or whitespace-only text: No keywords extracted")
            return []

        doc = spacy_nlp(text)
        keywords = [chunk.text for chunk in doc.noun_chunks]

        if not keywords:
            logging.warning("No keywords extracted from the text")

        return keywords
    except Exception as e:
        logging.error("Error extracting keywords:", exc_info=e)
        return []
def test_extract_keywords():
    text = "This is a test text for keyword extraction."
    keywords = extract_keywords(text)

    assert keywords is not None, "Keyword extraction failed"
    assert len(keywords) > 0, "No keywords extracted"
    print("Keyword extraction test passed")

test_extract_keywords()
def generate_thumbnails(frames, keywords, num_thumbnails=3):
    try:
        thumbnails = []
        for frame in frames:
            for _ in range(num_thumbnails):
                prompt = "A visually striking image of " + ", ".join(keywords)
                generated_image = stable_diffusion_pipeline(prompt, init_image=frame).images[0]
                thumbnails.append(generated_image)
        return thumbnails
    except Exception as e:
        logging.exception("Error generating thumbnails:", exc_info=e)
        return None
def process_video(video):
    try:
        # Determine the video path based on the type of input
        video_path = video.name if hasattr(video, 'name') else video

        # Extract Keyframes
        frames = extract_keyframes(video_path)
        if frames is None:
            return handle_error("Error extracting keyframes. Please check the video file.")

        # Transcribe Audio
        transcription = transcribe_audio(video_path)
        if transcription is None:
            return handle_error("Error transcribing audio. Please check the audio quality.")

        # Extract Keywords
        keywords = extract_keywords(transcription)
        if not keywords:
            return handle_error("Error extracting keywords. Please check the transcription.")

        # Use the first keyword as title, the full transcription as text, and a generic text placement description
        title = keywords[0] if keywords else "Thumbnail"
        text = transcription
        text_placement = "white letter center at bottom, modern and dynamic"

        # Generate Thumbnails
        thumbnail_images = generate_thumbnails(frames, keywords)
        if not thumbnail_images:
            return handle_error("Error generating thumbnails. Please try again later.")

        return thumbnail_images, "Thumbnails generated successfully."
    except Exception as e:
        logging.exception("Unexpected error:", exc_info=e)
        return handle_error("An unexpected error occurred. Please try again later.")
def handle_error(error_message):
    # Return a placeholder image and the error message
    placeholder = Image.new('RGB', (512, 512), color = (255, 0, 0))  # Placeholder image (red square)
    return [placeholder], error_message
# Gradio interface
interface = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Video"),
    outputs=[
        gr.Gallery(label="Generated Thumbnails"),
        gr.Textbox(label="Status", lines=2, placeholder="Status message will appear here...")
    ],
    title="YouTube Thumbnail Generator",
    description="Upload a video and generate multiple thumbnails using the video content and transcription.",
    live=True
)

interface.launch()
