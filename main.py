from fastapi import FastAPI, Response, UploadFile, File
from pydantic import BaseModel
from voice_rag_functions import (
    init_chain,
    mentor_answer,
    transcribe_audio,
    autocomplete_if_needed,
    is_recipe_related,
    terminal_voice_chat, quick_voice_question
)
from elevenlabs_functions import speak_text_to_stream

# Initialize the Rosendale Method coaching system
print(" Initializing Culinary Mentor with Rosendale Method...")
client, retriever, memory, llm = init_chain()
app = FastAPI(title="Rosendale Method Culinary Coach", version="2.0")

class TextQuestion(BaseModel):
    question: str

@app.get("/")
def read_root():
    return {
        "message": "Rosendale Method Culinary Coach API", 
        "description": "Teaching cooking principles, not recipes",
        "version": "2.0"
    }

@app.post("/coach/")
async def culinary_coach(data: TextQuestion):
    """Text question → Text coaching response"""
    try:
        print(f" Student question: {data.question}")
        
        # Enhance short questions
        full_query = autocomplete_if_needed(data.question, memory, llm)
        
        # Validate cooking-related content
        if not is_recipe_related(full_query):
            return {"error": "Please ask about cooking techniques, methods, or culinary principles."}
        
        # Get coaching response
        answer, emotion_type = mentor_answer(full_query, retriever, memory, llm)
        
        print(f"Coach response: {answer}")
        print(f" Emotion: {emotion_type}")
        
        return {
            "question": data.question,
            "enhanced_question": full_query if full_query != data.question else None,
        
            "coaching_response": answer,
            "emotion": emotion_type
        }
        
    except Exception as e:
        import traceback
        print("ERROR:", traceback.format_exc())
        return {"error": str(e)}

@app.post("/coach_voice/")
async def culinary_coach_voice(data: TextQuestion):
    """Text question → Audio coaching response"""
    try:
        print(f" Voice coaching request: {data.question}")
        
        # Enhance short questions
        full_query = autocomplete_if_needed(data.question, memory, llm)
        
        # Validate cooking-related content
        if not is_recipe_related(full_query):
            error_msg = "Please ask about cooking techniques, methods, or culinary principles."
            audio_stream = speak_text_to_stream(error_msg)
            return Response(
                audio_stream.getvalue(),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=error.mp3"}
            )
        
        # Get coaching response
        answer, emotion_type = mentor_answer(full_query, retriever, memory, llm)
        
        print(f" Coach response: {answer}")
        print(f" Emotion: {emotion_type}")
        
        if not answer or len(answer.strip()) < 3:
            return {"error": "Could not generate coaching response."}
        1
        # Generate audio response
        audio_stream = speak_text_to_stream(answer)
        audio_bytes = audio_stream.getvalue()
        
        print(f" Audio bytes generated: {len(audio_bytes)}")
        
        if len(audio_bytes) == 0:
            return {"error": "Generated audio is empty."}
        
        return Response(
            audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=coaching_response.mp3"}
        )
        
    except Exception as e:
        import traceback
        print("VOICE ERROR:", traceback.format_exc())
        return {"error": str(e)}

@app.post("/coach_audio_input/")
async def coach_audio_input(file: UploadFile = File(...)):
    """Audio question → Audio coaching response (full voice interaction)"""
    try:
        print(" Processing audio coaching session...")
        
        # Read and transcribe audio
        audio_bytes = await file.read()
        transcription = transcribe_audio(audio_bytes, client)
        
        if not transcription:
            error_msg = "I couldn't understand your question clearly. Could you try again?"
            audio_stream = speak_text_to_stream(error_msg)
            return Response(
                audio_stream.getvalue(),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=clarification.mp3"}
            )
        
        print(f" Transcribed: {transcription}")
        
        # Enhance and validate question
        full_query = autocomplete_if_needed(transcription, memory, llm)
        
        if not is_recipe_related(full_query):
            error_msg = "I'm here to coach you on cooking techniques and culinary principles. What would you like to learn about cooking?"
            audio_stream = speak_text_to_stream(error_msg)
            return Response(
                audio_stream.getvalue(),
                media_type="audio/mpeg",
                headers={"Content-Disposition": "inline; filename=redirect.mp3"}
            )
        
        # Generate coaching response
        answer, emotion_type = mentor_answer(full_query, retriever, memory, llm)
        
        print(f" Coach response: {answer}")
        
        # Convert to audio
        audio_stream = speak_text_to_stream(answer)
        audio_bytes = audio_stream.getvalue()
        
        print(f" Response audio: {len(audio_bytes)} bytes")
        
        return Response(
            audio_bytes,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=coaching_session.mp3"}
        )
        
    except Exception as e:
        import traceback
        print("AUDIO SESSION ERROR:", traceback.format_exc())
        error_msg = "I'm having trouble processing your audio. Let's try again with your cooking question."
        audio_stream = speak_text_to_stream(error_msg)
        return Response(
            audio_stream.getvalue(),
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=error_recovery.mp3"}
        )

# Legacy endpoints for backward compatibility
@app.post("/qa_text_to_speech/")
async def qa_text_to_speech(data: TextQuestion):
    """Legacy endpoint - redirects to coach_voice"""
    return await culinary_coach_voice(data)


@app.post("/ask_audio_base64/")
async def ask_audio_base64(file: UploadFile = File(...)):
    """
    User sends audio (question) → Mentor answers with Base64 audio
    """
    audio_bytes = await file.read()
    transcription = transcribe_audio(audio_bytes, client)
 
    if not transcription:
        msg = "I couldn't understand clearly, please try again."
        audio_stream = speak_text_to_stream(msg)
    else:
        full_query = autocomplete_if_needed(transcription, memory, llm)
 
        if not is_recipe_related(full_query):
            msg = "I'm here to coach you on cooking techniques and culinary principles."
            audio_stream = speak_text_to_stream(msg)
        else:
            answer, emotion = mentor_answer(full_query, retriever, memory, llm)
            audio_stream = speak_text_to_stream(answer)
 
    audio_bytes = audio_stream.getvalue()
    audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
 
    return JSONResponse(content={
        "user_question": transcription if transcription else None,
        "audio_base64": audio_base64,
        "format": "mp3"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
