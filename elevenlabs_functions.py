import os
import io
import re
from elevenlabs import ElevenLabs  # Import the ElevenLabs class, not client
from dotenv import load_dotenv

load_dotenv()

# Initialize ElevenLabs client correctly
elevenlabs_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

def clean_text_for_speech(text):
    """Clean text for better speech synthesis"""
    # Remove markdown formatting
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    
    # Remove special characters that might cause issues
    text = re.sub(r'[🍳✨🔥🌟❗─━]', '', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Ensure proper sentence endings
    text = text.strip()
    if text and not text.endswith(('.', '!', '?')):
        text += '.'
    
    return text

# def speak_text_to_stream(text, voice_id="O483h7ZB7zKaA4JmK9Wv"):#previous voice 29-08-25
# def speak_text_to_stream(text, voice_id="Mo9SFAWCFwzIAmrZsCLd"):#v2 voice use
def speak_text_to_stream(text, voice_id="mEE6giLueLdSOaKRUws3"):#Thomas voice
    """Convert text to speech and return as BytesIO stream"""
    try:
        cleaned = clean_text_for_speech(text)
        print(f"🎧 Generating audio for: {cleaned[:50]}...")

        # Generate audio using ElevenLabs with correct client reference
        audio_generator = elevenlabs_client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_monolingual_v1",
            text=cleaned,
            voice_settings={
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.2,
                "use_speaker_boost": True
            }
        )

        # Handle generator output correctly
        if hasattr(audio_generator, "__iter__") and not isinstance(audio_generator, (bytes, bytearray)):
            # If it's a generator, collect all chunks
            audio_bytes = b"".join(audio_generator)
        else:
            # If it's already bytes, use directly
            audio_bytes = audio_generator

        print(f"✓ Audio generated: {len(audio_bytes)} bytes")
        return io.BytesIO(audio_bytes)

    except Exception as e:
        print(f"❌ TTS Error: {e}")
        # Return a minimal audio stream for error cases
        return io.BytesIO(b"")
