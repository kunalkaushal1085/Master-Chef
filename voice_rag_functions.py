import os, re, wave, numpy as np, pyaudio, threading, queue, time
from pathlib import Path
from dotenv import load_dotenv
from docx import Document
from openai import OpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredWordDocumentLoader, UnstructuredPDFLoader
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferWindowMemory
import random
import tempfile
import soundfile as sf


# Enhanced principles extraction - Updated for Rosendale Method
KEEP = [
    r"philosophy", r"approach", r"why", r"because", r"ensures", r"prevents",
    r"results", r"should", r"never", r"always", r"critical", r"clarity",
    r"depth", r"restraint", r"fundamentals", r"technique", r"principle",
    r"coaching", r"guidance", r"mindset", r"understanding", r"mastery",
    r"teaches", r"learns", r"develops", r"builds", r"foundation",
    r"rules", r"control", r"texture", r"safety", r"avoid", r"use",
    r"essential", r"proper", r"method", r"ideal", r"recommended",
    r"precise", r"consistency", r"focus on", r"goal is",
    # Rosendale Method specific patterns
    r"rosendale method", r"key insight", r"common mistake", r"pro tip",
    r"what separates", r"the difference between", r"real secret",
    r"professional approach", r"kitchen wisdom", r"experience shows",
    r"problem solving", r"troubleshooting", r"when things go wrong",
    r"signature", r"technique breakdown", r"coaching moment"
]


EMOTIONAL_EXPRESSIONS = {
    "enthusiasm": [
        "I'm genuinely excited to share this principle with you!", 
        "This is where cooking mastery gets absolutely fascinating!",
        "Now we're diving into the real foundation of great cooking!",
        "This coaching moment is honestly one of my favorites!",
        "Here's where the Rosendale Method really shines!"
    ],
    "concern": [
        "I really want you to understand this crucial principle...",
        "This technique is critically important to master—",
        "Let me coach you through this because it truly matters—",
        "Please focus on this fundamental concept:",
        "Here's what genuinely concerns me when cooks skip this principle:"
    ],
    "pride": [
        "When you master this technique, you'll feel the difference immediately!",
        "There's honestly nothing like the confidence this principle builds...",
        "You'll know you've truly understood this when...",
        "This mastery is what really separates good cooks from exceptional ones!",
        "This is the foundation of professional cooking!"
    ],
    "empathy": [
        "I completely understand this principle can feel complex at first...",
        "Don't worry at all, every great cook has struggled with this concept—",
        "I remember feeling exactly the same way when learning this technique...",
        "It's totally normal to feel uncertain about this method...",
        "Every professional goes through this exact learning curve..."
    ],
    "curiosity": [
        "Have you ever genuinely wondered why this technique works...?",
        "What absolutely fascinates me about this method is...",
        "Think about this principle for just a moment—",
        "Here's something that might really surprise you about this technique:",
        "I'm genuinely curious—have you ever noticed this happening...?"
    ],
    "wisdom": [
        "Years in professional kitchens have taught me this principle...",
        "Here's what real experience has shown me about this technique:",
        "The deeper truth I've discovered through the Rosendale Method is...",
        "What I've learned through countless hours coaching is...",
        "After decades in the kitchen, this principle has proven essential..."
    ]
}


def extract_principles_from_pdf(src: str, dst: str):
    """Extract coaching principles from PDF documents"""
    if Path(dst).exists():
        return
    
    loader = UnstructuredPDFLoader(src)
    docs = loader.load()
    doc_out = Document()
    
    for doc in docs:
        text = doc.page_content
        paragraphs = text.split('\n')
        
        for para in paragraphs:
            para = para.strip()
            if para and any(re.search(pat, para, re.I) for pat in KEEP):
                para = re.sub(r'\s+', ' ', para)
                if len(para) > 20:
                    doc_out.add_paragraph(para)
    
    doc_out.save(dst)
    print(f"✓ PDF principles extracted → {dst}")


def recipe_to_principles(src: str, dst: str):
    """Extract coaching principles from Word documents"""
    if Path(dst).exists():
        return
    
    if not Path(src).exists():
        print(f"Warning: {src} not found, skipping...")
        return
        
    doc_in, doc_out = Document(src), Document()
    for p in doc_in.paragraphs:
        t = p.text.strip()
        if t and any(re.search(pat, t, re.I) for pat in KEEP):
            doc_out.add_paragraph(t)
    doc_out.save(dst)
    print(f"✓ Word doc principles extracted → {dst}")


def get_emotional_opener(context_keywords):
    """Enhanced emotional context detection for coaching"""
    context_lower = context_keywords.lower()
    
    if any(word in context_lower for word in ["safety", "dangerous", "avoid", "never", "mistake"]):
        return random.choice(EMOTIONAL_EXPRESSIONS["concern"]), "concern"
    elif any(word in context_lower for word in ["master", "perfect", "technique", "skill", "professional"]):
        return random.choice(EMOTIONAL_EXPRESSIONS["pride"]), "pride"
    elif any(word in context_lower for word in ["why", "science", "understand", "fascinating", "principle"]):
        return random.choice(EMOTIONAL_EXPRESSIONS["curiosity"]), "curiosity"
    elif any(word in context_lower for word in ["difficult", "hard", "struggle", "problem", "troubleshoot"]):
        return random.choice(EMOTIONAL_EXPRESSIONS["empathy"]), "empathy"
    elif any(word in context_lower for word in ["experience", "learned", "years", "wisdom", "rosendale"]):
        return random.choice(EMOTIONAL_EXPRESSIONS["wisdom"]), "wisdom"
    else:
        return random.choice(EMOTIONAL_EXPRESSIONS["enthusiasm"]), "enthusiasm"


def init_chain():
    """Initialize the Rosendale Method coaching chain"""
    load_dotenv()
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Extract principles from available documents
    recipe_to_principles("48 Hour Short Ribs.docx", "Rosendale_Principles.docx")
    extract_principles_from_pdf("Sous-Vide-Fish-1.pdf", "Fish_Principles.docx")
    
    all_docs = []
    
    # Load available principle documents
    for doc_path in ["Rosendale_Principles.docx", "Fish_Principles.docx"]:
        if Path(doc_path).exists():
            docs = UnstructuredWordDocumentLoader(doc_path).load()
            all_docs.extend(docs)
            print(f"✓ Loaded {len(docs)} documents from {doc_path}")
    
    if not all_docs:
        print("Warning: No principle documents found!")
        return openai_client, None, None, None
    
    # Optimized chunking for coaching principles
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
    split_docs = splitter.split_documents(all_docs)
    print(f"✓ Split into {len(split_docs)} principle chunks")

    # Create vector store for principle retrieval
    vectordb = Chroma.from_documents(
        split_docs, 
        OpenAIEmbeddings(), 
        persist_directory="./chroma_rosendale"
    )   
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})


    # Conversation memory for coaching context
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        return_messages=True, 
        k=3  # Keep recent context for coaching flow
    )
    
    # Optimized LLM for coaching responses
    llm = ChatOpenAI(
        model="gpt-4o", 
        temperature=0.7, 
        max_tokens=350  # Concise coaching responses
    )
    
    print("✓ Rosendale Method coaching chain initialized")
    return openai_client, retriever, memory, llm


def mentor_answer(q, retriever, memory, llm):
    """Generate detailed conversational coaching responses using Rosendale Method"""
    if not retriever:
        return ("I'm still getting familiar with your cooking principles, but I'm excited to start coaching you! Every great cook begins with curiosity about technique. What specific cooking challenge would you like to explore together?", "empathy")
    
    # Use the new invoke method
    ctx_blocks = retriever.invoke(q)
    if not ctx_blocks:
        response = ("I love diving into cooking principles with passionate cooks like yourself! "
                   "There's so much wisdom to share about technique, timing, and the science behind great food. "
                   "What culinary challenge has been on your mind lately? I'm here to help you master it.")
        return response, "enthusiasm"
    
    # Prepare context from principles
    context = "\n".join(d.page_content for d in ctx_blocks[:3])[:1800]
    
    # Get conversation history
    history_messages = memory.chat_memory.messages[-6:]
    history = "\n".join(f"{m.type.upper()}: {m.content}" for m in history_messages)[:800]
    
    # Get emotion type without the opener text
    _, emotion_type = get_emotional_opener(context + " " + q)
    
    # Enhanced coaching prompt WITHOUT emotional opener injection
    prompt = (
        f"COACHING CONTEXT: {context}\n"
        f"CONVERSATION HISTORY: {history}\n"
        f"STUDENT QUESTION: {q}\n\n"
        
        "You are an experienced culinary coach using the Rosendale Method. "
        "Your responses should be conversational, warm, and detailed enough to truly help students understand. "
        
        "RESPONSE GUIDELINES:\n"
        "- Be conversational and engaging, like talking to a friend in the kitchen\n"
        "- Explain WHY techniques work with genuine enthusiasm and detail\n"
        "- Share professional insights and problem-solving wisdom\n"
        "- Use specific examples to illustrate principles, but teach concepts, not procedures\n"
        "- Show your passion for cooking through your explanations\n"
        "- Make it feel like a mentoring conversation, not a lecture\n"
        "- Length: 4-6 sentences that flow naturally together\n"
        "- Start directly with your teaching point - avoid filler words like 'Ah,' 'Oh,' etc.\n"
        "- End with an engaging question that encourages deeper exploration\n"
        "- Never provide step-by-step recipes or mention specific chef names\n"
        "- Focus on building understanding and confidence\n"
        
        "Remember: You're building a cook's intuition through clear, direct teaching."
    )
    
    try:
        response = llm.invoke(prompt).content.strip()
        
        # Clean up any remaining filler expressions
        response = re.sub(r"^(Ah+|Oh+|Um+|Well+),?\s*", '', response, flags=re.I)
        response = re.sub(r"^(You know|Listen|Look),?\s*", '', response, flags=re.I)
        
        # Ensure proper ending punctuation
        if not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Update conversation memory
        memory.chat_memory.add_user_message(q)
        memory.chat_memory.add_ai_message(response)
        
        return response, emotion_type
        
    except Exception as e:
        print(f"Error in mentor_answer: {e}")
        return ("That's a fascinating question that really gets to the heart of cooking mastery. Let me think about the best way to break this down for you. What specific aspect of this technique has been challenging you the most?", "empathy")


def is_speech(audio_data, threshold=0.005):
    """Check if audio contains speech"""
    rms = np.sqrt(np.mean(np.square(audio_data)))
    return rms > threshold


def is_valid_transcription(text):
    """Validate transcription quality"""
    meaningless = {
        "thank you", "thank you for watching", "hello", "hi", "...", "", "welcome back",
        "thank you everyone", "thanks for watching", "okay", "ok", "yes", "no", "nothing",
        "Thank you for watching!", "Thank you for watching", " Thank you for watching."
    }
    cleaned = text.strip().lower()
    non_latin_ratio = sum(1 for c in cleaned if not re.match(r'[a-zA-Z0-9\s,.!?]', c)) / max(len(cleaned), 1)
    return (
        len(cleaned) > 5 and
        cleaned not in meaningless and
        non_latin_ratio < 0.3
    )


def transcribe_audio(audio_bytes, client):
    """Transcribe audio to text using OpenAI Whisper"""
    # Create temporary file for audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        temp_audio.write(audio_bytes)
        audio_path = temp_audio.name

    try:
        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",  # Fixed model name
                file=audio_file,
                response_format="text",
                language="en",
                temperature=0.2,
                prompt="Listen for cooking and culinary questions about techniques, methods, and principles."
            )
        os.remove(audio_path)
        
        if not is_valid_transcription(transcription):
            return None
        return transcription
        
    except Exception as e:
        print("Transcription error:", e)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None


def autocomplete_if_needed(user_input, memory, llm):
    """Enhance short queries for better coaching context"""
    if len(user_input.split()) < 4:
        history = memory.chat_memory.messages[-6:]
        chat_history_str = "\n".join(f"{m.type.upper()}: {m.content}" for m in history)
        prompt = (
            f"The student asked: '{user_input}'\n"
            f"Coaching conversation so far:\n{chat_history_str}\n"
            f"Assume they're asking about cooking techniques or culinary principles. "
            f"What is the most likely complete question they meant to ask about cooking mastery?"
        )
        try:
            response = llm.invoke(prompt)
            return response.content.strip()
        except:
            return user_input
    return user_input


def is_recipe_related(query):
    """Check if query is related to cooking/culinary topics"""
    cooking_keywords = [
        "recipe", "cook", "bake", "grill", "roast", "sauté", "fry", "steam", "boil",
        "sauce", "garlic", "onion", "burn", "temperature", "heat", "pan", "oil",
        "technique", "method", "flavor", "texture", "ingredient", "season", "salt"
    ]
    
    # Also check for common cooking question patterns
    cooking_patterns = [
        r"why does.*burn", r"how do.*cook", r"what.*temperature",
        r"how.*prevent", r"why.*tough", r"how.*make", r"why.*dry"
    ]
    
    query_lower = query.lower()
    
    # Check keywords
    if any(word in query_lower for word in cooking_keywords):
        return True
    
    # Check patterns
    if any(re.search(pattern, query_lower) for pattern in cooking_patterns):
        return True
        
    return False


def get_qa_answer(full_query, qa_chain):
    """Legacy function for backward compatibility"""
    response = qa_chain.invoke({"question": full_query})
    return response["answer"]
