import os
import sys
import base64
import requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template
import speech_recognition as sr
from PIL import Image
from io import BytesIO
from flask_cors import CORS

# Ensure UTF-8 output for proper character handling
sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()
API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("LLM_MODEL", "x-ai/grok-4-fast:free")
BASE_URL = "https://openrouter.ai/api/v1/chat/completions"

app = Flask(__name__)
CORS(app) 

def ask_multimodal_model(text_input: str, image_data: str = None):
    """Sends text and optional image data to a multimodal LLM."""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are an expert farming advisor. "
                "Answer in Malayalam if the question is in Malayalam. "
                "Give very short, concise advice using 3-5 bullet points. "
                "Avoid long paragraphs. If an image is provided, analyze it to give a better answer."
            )
        },
        {"role": "user", "content": text_input}
    ]

    if image_data:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}}
            ]
        })

    payload = {
        "model": MODEL_NAME,
        "messages": messages
    }

    try:
        response = requests.post(BASE_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        if "choices" in result:
            return result["choices"][0]["message"]["content"].strip()
        else:
            return f"❌ Error: Unexpected response {result}"
    except requests.exceptions.RequestException as e:
        return f"❌ Error: {str(e)}"

@app.route('/')
def index():
    return "AI Farming Advisor Backend is running!"

@app.route('/query-image', methods=['POST'])
def query_with_image():
    try:
        data = request.form.get('question')
        image_file = request.files.get('image')

        image_b64 = None
        if image_file:
            img = Image.open(image_file.stream)
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            image_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        answer = ask_multimodal_model(data, image_b64)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/query-voice', methods=['POST'])
def query_with_voice():
    try:
        audio_file = request.files.get('audio')
        if not audio_file:
            return jsonify({"error": "No audio file provided"}), 400

        recognizer = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                question_text = recognizer.recognize_google(audio_data, language='ml-IN')
            except sr.UnknownValueError:
                return jsonify({"error": "Could not understand audio"}), 400
            except sr.RequestError as e:
                return jsonify({"error": f"Speech recognition service error: {e}"}), 500

        answer = ask_multimodal_model(question_text)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)