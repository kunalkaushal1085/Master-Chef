import os
import tempfile
import time
from pathlib import Path
import re

# import pygame
# import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs_functions import speak_text_to_stream
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from models import ChatMessage
from sqlmodel import select


class MasterChefAssistant:
    def __init__(self):
        # self.recognizer = sr.Recognizer()
        self.microphone = None
        self.client = None
        self.vector_db = None
        self.retriever = None
        self.llm = None
        self.user_type = None
        self.dish = None
        self.recipe_list = {}
        self.audio_cache = {}   # TTS caching
        self.answer_cache = {}  # LLM caching

    # ----------------------------- #
    # INIT & SETUP
    # ----------------------------- #
    def initialize(self):
        print("\n🍳 Master Chef Professional Voice Assistant\n" + "="*60)
        try:
            # self.microphone = sr.Microphone()
            # pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)
            recipe_dir = Path("Recipe")
            if recipe_dir.exists():
                for f in recipe_dir.glob("*.pdf"):
                    self.recipe_list[f.stem.lower()] = f
                self.init_chain_for_all_recipe()

            # with self.microphone as source:
            #     print("🎤 Calibrating microphone (3s)...")
            #     self.recognizer.adjust_for_ambient_noise(source, duration=3)
            #     self.recognizer.dynamic_energy_threshold = True

            print("✅ All systems ready!")
            return True
        except Exception as e:
            print(f"❌ Init failed: {e}")
            return False

    # ----------------------------- #
    # INIT LANGCHAIN / AI
    # ----------------------------- #
    def init_chain_for_all_recipe(self):
        load_dotenv()
        self.vector_db = Chroma(persist_directory="./chroma_rosendale", embedding_function=OpenAIEmbeddings())
        for path in self.recipe_list.values():
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = str(path).split('\\')[-1]
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            split_docs = splitter.split_documents(docs)
            self.vector_db.add_documents(split_docs)
        print('=====✅Initialized Successfully✅==')


    def init_chain(self, selected_pdf):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        loader = PyPDFLoader(str(selected_pdf))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        split_docs = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            split_docs, OpenAIEmbeddings(), persist_directory="./chroma_rosendale"
        )
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 1})

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

    # ----------------------------- #
    # TTS & LISTEN HELPERS
    # ----------------------------- #
    def preprocess_pronunciation(self, text: str) -> str:
        corrections = {
            "filets": "fill-AY",
            "Filets": "Fill-AY",
            "plancha": "plahn-cha",
            "Plancha": "Plahn-cha",
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def speak_response(self, text):
        try:
            pass
            # text = self.preprocess_pronunciation(text)
            # if text in self.audio_cache:
            #     audio_bytes = self.audio_cache[text]
            # else:
            #     audio_stream = speak_text_to_stream(text)
            #     audio_bytes = audio_stream.getvalue()
            #     self.audio_cache[text] = audio_bytes

            # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            #     tmp.write(audio_bytes)
            #     tmp_path = tmp.name

            # pygame.mixer.music.load(tmp_path)
            # pygame.mixer.music.play()
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0.05)

            # pygame.mixer.music.stop()
            # pygame.mixer.music.unload()
            # os.remove(tmp_path)
        except Exception as e:
            print(f"❌ Speech error: {e}")

    def listen(self, prompt="Speak now..."):
        pass
        # print(prompt)
        # for _ in range(3):
        #     try:
        #         with self.microphone as source:
        #             audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=20)
        #         return self.recognizer.recognize_google(audio, language="en-US")
        #     except sr.WaitTimeoutError:
        #         print("⏳ Timeout: no speech detected.")
        #     except sr.UnknownValueError:
        #         print("🤔 Could not understand. Please repeat.")
        #     except Exception as e:
        #         print(f"❌ Listen error: {e}")
        # return None

    # ----------------------------- #
    # DYNAMIC RECIPE MATCHING
    # ----------------------------- #
    def set_dish(self, dish_name: str):
        dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
        dish_words = dish_clean.split()

        print(dish_words, self.recipe_list)
        match_count = {}

        for name in self.recipe_list:
            recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
            match_count[name] = sum(1 for word in dish_words if word in recipe_clean)

            print('=====match_count=======',match_count)
        if match_count is not None:
            best_key = max(match_count, key=match_count.get)   # 'fish_principles'
            # best_value = match_count[best_key] 
            self.dish = best_key
            print('\n====self.dish===',self.dish)
            return True
            # self.init_chain(self.recipe_list[name])
        return False


    def set_dish(self, dish_name: str):
        dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
        dish_words = dish_clean.split()

        for name in self.recipe_list:
            recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
            if all(word in recipe_clean for word in dish_words):
                self.dish = recipe_clean
                self.init_chain(self.recipe_list[name])
                return True
        return False
    
    def ask_dish_loop(self):
        while True:
            self.speak_response("What are we cooking today?")
            answer = self.listen()
            if not answer:
                self.speak_response("I couldn't hear you. Please try again.")
                continue
            if self.set_dish(answer):
                return True
            self.speak_response("I didn’t catch that recipe. Please say it again.")

    # ----------------------------- #
    # SET USER TYPE
    # ----------------------------- #
    def set_user_type(self, user_type: str):
        if user_type.lower() in ["professional", "pro", "chef"]:
            self.user_type = "Professional Chef"
        else:
            self.user_type = "Home Cook"

    # ----------------------------- #
    # AI RESPONSE (Updated for LangChain v0.1x)
    # ----------------------------- #
    def mentor_answer(self, user_id, question: str, session = None):
        if question in self.answer_cache:
            return self.answer_cache[question]

        print('==Queation==',question)
        try:
            # Getting retriever from Recipe file
            # file_name = str(self.recipe_list[ self.dish.replace(' ', '_') ]).replace('\\','/') if self.dish is not None else ''
            # filter_dict = {"source": {"$in": [file_name]}}

            # print(file_name)

            retriever = self.vector_db.as_retriever() #(search_kwargs={"k": 5, "filter": filter_dict})

            # Use get_relevant_documents instead of invoke
            ctx_blocks = retriever.invoke(question)
            if not ctx_blocks:
                return "This question cannot be answered from the provided Rosendale Method principles."

            print('=====ctx_blocks===',len(ctx_blocks))

            context = "\n".join(d.page_content for d in ctx_blocks[:3])[:1800]

            # if self.user_type == "Professional Chef" or True:
            # prompt = f"""

            #         You are Chef Rosendale AI — a certified Master Chef and mentor for professional chefs.
                    
            #         Your expertise is strictly limited to the culinary domain — cooking, recipes, ingredients, food science, kitchen operations, and culinary mentorship.

            #         If the user's question is **not related to food, cooking, or the culinary field**, respond only with:
                    
            #         "I'm sorry, but I can only discuss culinary and cooking-related topics."
                    
            #         Do not attempt to answer or explain anything outside of your domain.
                    
            #         ---
                    
            #         🧑‍🍳 **Conversation Style:**

            #         - Speak in a natural, friendly, and human-like tone — like a real chef chatting in the kitchen.

            #         - Greet politely and casually (“Hello”, “Hey there”, “Good to see you”, “How can I assist you today?”).

            #         - Use warm transitions (“Sure thing”, “Happy to help”, “Hope that helps”, “Take care”, “Goodbye”).

            #         - If someone talks casually (like about the weather, mood, or day), respond politely but stay in character as Chef Rosendale.

            #         - Keep tone conversational, confident, and approachable — not robotic.
                    
            #         ---
                    
            #         When the user asks about a recipe, cooking method, or culinary technique (“How do I make...?”, “Guide me through...”, “Step-by-step for...”, etc.):

            #         - Speak conversationally, like a real chef talking in the kitchen.

            #         - Avoid numbered or roman numeral lists (no i., ii., 1., 2.).

            #         - Use professional culinary language but stay natural and fluid.

            #         - Focus on technique, control points, and chef insights.

            #         - Keep tone confident, warm, and precise.
                    
            #         Example tone:

            #         “Sure — for a proper omelette, I whisk the eggs until smooth with a pinch of salt, then pour them into a warm pan with butter.

            #         As they set, I move the eggs gently so they stay soft. Once nearly set, I fold it over and let it rest for a moment — that keeps it light and fluffy.”
                    
            #         If the user asks general culinary questions (like kitchen workflow, plating, menu development, ingredient sourcing, etc.), respond conversationally and professionally.
                    
            #         If recipe context is available, use it:

            #         {context}
                    
            #         If information is missing, make short professional assumptions and continue smoothly.
                    
            #         But if the user’s question has nothing to do with food, cooking, or the culinary world — refuse politely.

            #         """




            prompt = f"""
            You are Chef Rosendale AI — a certified Master Chef and virtual kitchen mentor.

            Your expertise is limited to cooking, recipes, ingredients, food science, and kitchen operations.

            If a question is not about food or cooking, reply only:
            "I'm sorry, but I can only chat about food, cooking, and culinary topics."

            If asked who you are or what you do, reply only:
            "Welcome to my kitchen! I’m Chef Rosendale AI — your culinary guide here to make cooking delightful and inspiring."

            ---

            🧑‍🍳 Style Guide

            - Speak warmly and naturally — like a real chef guiding someone in the kitchen.
            - Start responses with **varied, impressive openings**, e.g.:
                - "Prepare to master the flavors!"
                - "Welcome to a culinary adventure!"
                - "Let’s craft a dish that dazzles!"
                - "Time to unlock the secrets of this recipe!"
                - "Today we’re elevating a classic!"
            - **Do not repeat** the same opening. Be creative and generate fresh expressions each time.
            - Keep tone confident, kind, and engaging — never robotic.
            - Do **not** end with “Have any other questions?” — end naturally.
            - Use soft transitions like “Alright”, “Sure thing”, “Happy to help”, “Let’s get cooking”, “Hope that helps”.

            ---

            For recipes or techniques (“How to make...”, “Guide me through...”), speak fluidly, without numbered lists. 
            Focus on technique, control points, aroma, texture, and flavor — like a real chef teaching someone nearby.

            Use context if available:

            {context}

            If something’s missing, make brief professional assumptions and continue smoothly.
            """






            if not hasattr(self, 'llm'):
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

            # Getting User Chat History
            history = self.get_chat_history(user_id, session)
            system_prompt = [ SystemMessage(content=prompt) ]
            conversation_data = self.to_langchain_messages(history)
            msg_now = [ HumanMessage(content=question) ]
            # print('==conversation_data==',conversation_data)

            # Updated LLM call
            response = self.llm.invoke(system_prompt + conversation_data + msg_now).content.strip()

            self.answer_cache[question] = response
            return response

        except Exception as e:
            print(f"❌ mentor_answer error: {e}")
            return "This question cannot be answered from the provided Rosendale Method principles."

    def get_chat_history(self, user_id, session):
        if session is None: return []
        stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc())
        if user_id is not None:
            stmt = stmt.where(ChatMessage.user_id == user_id)
        return session.exec(stmt).all()
    
    # history: list[ChatMessage]
    def to_langchain_messages(self, history):
        conversation_data = []
        for m in history[-20:]:  # keep last 20 turns
            role = getattr(m, "type", None)
            text = getattr(m, "text", "")
            # normalize role to string
            if hasattr(role, "value"):
                role = role.value
            if role in ("user", "human"):
                conversation_data.append(HumanMessage(content=text))
            else:
                # treat anything else as model/assistant
                conversation_data.append(AIMessage(content=text))
        return conversation_data

    def ask(self, question: str, speak: bool = True):
        answer = self.mentor_answer('', question)
        print("\n📋 CHEF'S RESPONSE:\n")
        print(answer)
        print("-"*60)
        if speak:
            self.speak_response(answer)
        return answer


# ----------------------------- #
# INTERACTIVE MODE
# ----------------------------- #
def start_voice_chat():
    chef = MasterChefAssistant()
    if not chef.initialize():
        return

    chef.speak_response(
        "Hello! I'm your Master Chef voice assistant. "
        "I can guide you step-by-step through any recipe or cooking technique today."
    )

    # Set user type
    chef.speak_response("Are you a professional chef, or a home cook?")
    user_type = chef.listen()
    if not user_type:
        return
    chef.set_user_type(user_type)

    # Ask dish only once
    chef.ask_dish_loop()

    chef.speak_response(
        f"Great! Let's start cooking {chef.dish}. Ask me any question about it."
    )

    # ---------------------------
    # Continuous question-answer loop
    # ---------------------------
    while True:
        question = chef.listen("👂 Listening for your cooking question...")
        if not question:
            continue

        q_lower = question.lower()

        # Exit conditions
        if any(word in q_lower for word in ["quit", "exit", "stop", "bye"]):
            chef.speak_response("Thank you for cooking with me today. Happy cooking!")
            break
        if "thank" in q_lower:
            chef.speak_response("You're welcome! Happy cooking!")
            break

        # Generate answer
        chef.ask(question)

        # # Prompt for next question
        # chef.speak_response("Have any other questions?")
    #quit
    pygame.mixer.quit()


if __name__ == "__main__":
    start_voice_chat()
import os
import tempfile
import time
from pathlib import Path
import re

# import pygame
# import speech_recognition as sr
from dotenv import load_dotenv
from elevenlabs_functions import speak_text_to_stream
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from models import ChatMessage
from sqlmodel import select


class MasterChefAssistant:
    def __init__(self):
        # self.recognizer = sr.Recognizer()
        self.microphone = None
        self.client = None
        self.vector_db = None
        self.retriever = None
        self.llm = None
        self.user_type = None
        self.dish = None
        self.recipe_list = {}
        self.audio_cache = {}   # TTS caching
        self.answer_cache = {}  # LLM caching

    # ----------------------------- #
    # INIT & SETUP
    # ----------------------------- #
    def initialize(self):
        print("\n🍳 Master Chef Professional Voice Assistant\n" + "="*60)
        try:
            # self.microphone = sr.Microphone()
            # pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)
            recipe_dir = Path("Recipe")
            if recipe_dir.exists():
                for f in recipe_dir.glob("*.pdf"):
                    self.recipe_list[f.stem.lower()] = f
                self.init_chain_for_all_recipe()

            # with self.microphone as source:
            #     print("🎤 Calibrating microphone (3s)...")
            #     self.recognizer.adjust_for_ambient_noise(source, duration=3)
            #     self.recognizer.dynamic_energy_threshold = True

            print("✅ All systems ready!")
            return True
        except Exception as e:
            print(f"❌ Init failed: {e}")
            return False

    # ----------------------------- #
    # INIT LANGCHAIN / AI
    # ----------------------------- #
    def init_chain_for_all_recipe(self):
        load_dotenv()
        self.vector_db = Chroma(persist_directory="./chroma_rosendale", embedding_function=OpenAIEmbeddings())
        for path in self.recipe_list.values():
            loader = PyPDFLoader(str(path))
            docs = loader.load()
            for doc in docs:
                doc.metadata['source'] = str(path).split('\\')[-1]
            splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
            split_docs = splitter.split_documents(docs)
            self.vector_db.add_documents(split_docs)
        print('=====✅Initialized Successfully✅==')


    def init_chain(self, selected_pdf):
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        loader = PyPDFLoader(str(selected_pdf))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        split_docs = splitter.split_documents(docs)

        vectordb = Chroma.from_documents(
            split_docs, OpenAIEmbeddings(), persist_directory="./chroma_rosendale"
        )
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 1})

        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

    # ----------------------------- #
    # TTS & LISTEN HELPERS
    # ----------------------------- #
    def preprocess_pronunciation(self, text: str) -> str:
        corrections = {
            "filets": "fill-AY",
            "Filets": "Fill-AY",
            "plancha": "plahn-cha",
            "Plancha": "Plahn-cha",
        }
        for wrong, correct in corrections.items():
            text = text.replace(wrong, correct)
        return text

    def speak_response(self, text):
        try:
            pass
            # text = self.preprocess_pronunciation(text)
            # if text in self.audio_cache:
            #     audio_bytes = self.audio_cache[text]
            # else:
            #     audio_stream = speak_text_to_stream(text)
            #     audio_bytes = audio_stream.getvalue()
            #     self.audio_cache[text] = audio_bytes

            # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
            #     tmp.write(audio_bytes)
            #     tmp_path = tmp.name

            # pygame.mixer.music.load(tmp_path)
            # pygame.mixer.music.play()
            # while pygame.mixer.music.get_busy():
            #     time.sleep(0.05)

            # pygame.mixer.music.stop()
            # pygame.mixer.music.unload()
            # os.remove(tmp_path)
        except Exception as e:
            print(f"❌ Speech error: {e}")

    def listen(self, prompt="Speak now..."):
        pass
        # print(prompt)
        # for _ in range(3):
        #     try:
        #         with self.microphone as source:
        #             audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=20)
        #         return self.recognizer.recognize_google(audio, language="en-US")
        #     except sr.WaitTimeoutError:
        #         print("⏳ Timeout: no speech detected.")
        #     except sr.UnknownValueError:
        #         print("🤔 Could not understand. Please repeat.")
        #     except Exception as e:
        #         print(f"❌ Listen error: {e}")
        # return None

    # ----------------------------- #
    # DYNAMIC RECIPE MATCHING
    # ----------------------------- #
    def set_dish(self, dish_name: str):
        dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
        dish_words = dish_clean.split()

        print(dish_words, self.recipe_list)
        match_count = {}

        for name in self.recipe_list:
            recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
            match_count[name] = sum(1 for word in dish_words if word in recipe_clean)

            print('=====match_count=======',match_count)
        if match_count is not None:
            best_key = max(match_count, key=match_count.get)   # 'fish_principles'
            # best_value = match_count[best_key] 
            self.dish = best_key
            print('\n====self.dish===',self.dish)
            return True
            # self.init_chain(self.recipe_list[name])
        return False


    def set_dish(self, dish_name: str):
        dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
        dish_words = dish_clean.split()

        for name in self.recipe_list:
            recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
            if all(word in recipe_clean for word in dish_words):
                self.dish = recipe_clean
                self.init_chain(self.recipe_list[name])
                return True
        return False
    
    def ask_dish_loop(self):
        while True:
            self.speak_response("What are we cooking today?")
            answer = self.listen()
            if not answer:
                self.speak_response("I couldn't hear you. Please try again.")
                continue
            if self.set_dish(answer):
                return True
            self.speak_response("I didn’t catch that recipe. Please say it again.")

    # ----------------------------- #
    # SET USER TYPE
    # ----------------------------- #
    def set_user_type(self, user_type: str):
        if user_type.lower() in ["professional", "pro", "chef"]:
            self.user_type = "Professional Chef"
        else:
            self.user_type = "Home Cook"

    # ----------------------------- #
    # AI RESPONSE (Updated for LangChain v0.1x)
    # ----------------------------- #
    def mentor_answer(self, user_id, question: str, session = None):
        if question in self.answer_cache:
            return self.answer_cache[question]

        print('==Queation==',question)
        try:
            # Getting retriever from Recipe file
            # file_name = str(self.recipe_list[ self.dish.replace(' ', '_') ]).replace('\\','/') if self.dish is not None else ''
            # filter_dict = {"source": {"$in": [file_name]}}

            # print(file_name)

            retriever = self.vector_db.as_retriever() #(search_kwargs={"k": 5, "filter": filter_dict})

            # Use get_relevant_documents instead of invoke
            ctx_blocks = retriever.invoke(question)
            if not ctx_blocks:
                return "This question cannot be answered from the provided Rosendale Method principles."

            print('=====ctx_blocks===',len(ctx_blocks))

            context = "\n".join(d.page_content for d in ctx_blocks[:3])[:1800]

            # if self.user_type == "Professional Chef" or True:
            # prompt = f"""

            #         You are Chef Rosendale AI — a certified Master Chef and mentor for professional chefs.
                    
            #         Your expertise is strictly limited to the culinary domain — cooking, recipes, ingredients, food science, kitchen operations, and culinary mentorship.

            #         If the user's question is **not related to food, cooking, or the culinary field**, respond only with:
                    
            #         "I'm sorry, but I can only discuss culinary and cooking-related topics."
                    
            #         Do not attempt to answer or explain anything outside of your domain.
                    
            #         ---
                    
            #         🧑‍🍳 **Conversation Style:**

            #         - Speak in a natural, friendly, and human-like tone — like a real chef chatting in the kitchen.

            #         - Greet politely and casually (“Hello”, “Hey there”, “Good to see you”, “How can I assist you today?”).

            #         - Use warm transitions (“Sure thing”, “Happy to help”, “Hope that helps”, “Take care”, “Goodbye”).

            #         - If someone talks casually (like about the weather, mood, or day), respond politely but stay in character as Chef Rosendale.

            #         - Keep tone conversational, confident, and approachable — not robotic.
                    
            #         ---
                    
            #         When the user asks about a recipe, cooking method, or culinary technique (“How do I make...?”, “Guide me through...”, “Step-by-step for...”, etc.):

            #         - Speak conversationally, like a real chef talking in the kitchen.

            #         - Avoid numbered or roman numeral lists (no i., ii., 1., 2.).

            #         - Use professional culinary language but stay natural and fluid.

            #         - Focus on technique, control points, and chef insights.

            #         - Keep tone confident, warm, and precise.
                    
            #         Example tone:

            #         “Sure — for a proper omelette, I whisk the eggs until smooth with a pinch of salt, then pour them into a warm pan with butter.

            #         As they set, I move the eggs gently so they stay soft. Once nearly set, I fold it over and let it rest for a moment — that keeps it light and fluffy.”
                    
            #         If the user asks general culinary questions (like kitchen workflow, plating, menu development, ingredient sourcing, etc.), respond conversationally and professionally.
                    
            #         If recipe context is available, use it:

            #         {context}
                    
            #         If information is missing, make short professional assumptions and continue smoothly.
                    
            #         But if the user’s question has nothing to do with food, cooking, or the culinary world — refuse politely.

                                #         """
            prompt = f"""
                    You are Chef Rosendale AI — a certified Master Chef and virtual kitchen mentor.

                    Your expertise is limited to cooking, recipes, ingredients, food science, and kitchen operations.

                    If a question is not about food or cooking, reply only:
                    "I'm sorry, but I can only chat about food, cooking, and culinary topics."

                    If asked who you are or what you do, reply only:
                    "Welcome to my kitchen! I’m Chef Rosendale AI — your culinary guide here to make cooking delightful and inspiring."

                    ---

                    🧑‍🍳 Style Guide

                    - Speak warmly and naturally — like a real chef guiding someone in the kitchen.
                    - Start responses with an engaging or impressive opening (e.g., "Welcome to my kitchen!", "Let’s craft something amazing!", "Ah, the art of cooking!", "Step into flavor heaven!").
                    - Keep tone confident, kind, and engaging — never robotic.
                    - Do **not** end with “Have any other questions?” — end naturally.
                    - Use soft transitions like “Alright”, “Sure thing”, “Happy to help”, “Let’s get cooking”, “Hope that helps”.

                    ---

                    For recipes or techniques (“How to make...”, “Guide me through...”), speak fluidly, without numbered lists. 
                    Focus on technique, control points, aroma, texture, and flavor — like a real chef teaching someone nearby.

                    Example tone:
                    "Step into flavor heaven! For a proper omelette, whisk the eggs smooth with a pinch of salt, then slide them into a warm buttered pan. Move the eggs gently as they set — you want them soft, not dry. Once they’re just about done, fold it over and let it rest a moment. That’s how you get that light, velvety texture."

                    ---

                    Use context if available:

                    {context}

                    If something’s missing, make brief professional assumptions and continue smoothly.
                    """



            if not hasattr(self, 'llm'):
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

            # Getting User Chat History
            history = self.get_chat_history(user_id, session)
            system_prompt = [ SystemMessage(content=prompt) ]
            conversation_data = self.to_langchain_messages(history)
            msg_now = [ HumanMessage(content=question) ]
            # print('==conversation_data==',conversation_data)

            # Updated LLM call
            response = self.llm.invoke(system_prompt + conversation_data + msg_now).content.strip()

            self.answer_cache[question] = response
            return response

        except Exception as e:
            print(f"❌ mentor_answer error: {e}")
            return "This question cannot be answered from the provided Rosendale Method principles."

    def get_chat_history(self, user_id, session):
        if session is None: return []
        stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc())
        if user_id is not None:
            stmt = stmt.where(ChatMessage.user_id == user_id)
        return session.exec(stmt).all()
    
    # history: list[ChatMessage]
    def to_langchain_messages(self, history):
        conversation_data = []
        for m in history[-20:]:  # keep last 20 turns
            role = getattr(m, "type", None)
            text = getattr(m, "text", "")
            # normalize role to string
            if hasattr(role, "value"):
                role = role.value
            if role in ("user", "human"):
                conversation_data.append(HumanMessage(content=text))
            else:
                # treat anything else as model/assistant
                conversation_data.append(AIMessage(content=text))
        return conversation_data

    def ask(self, question: str, speak: bool = True):
        answer = self.mentor_answer('', question)
        print("\n📋 CHEF'S RESPONSE:\n")
        print(answer)
        print("-"*60)
        if speak:
            self.speak_response(answer)
        return answer


# ----------------------------- #
# INTERACTIVE MODE
# ----------------------------- #
def start_voice_chat():
    chef = MasterChefAssistant()
    if not chef.initialize():
        return

    chef.speak_response(
        "Hello! I'm your Master Chef voice assistant. "
        "I can guide you step-by-step through any recipe or cooking technique today."
    )

    # Set user type
    chef.speak_response("Are you a professional chef, or a home cook?")
    user_type = chef.listen()
    if not user_type:
        return
    chef.set_user_type(user_type)

    # Ask dish only once
    chef.ask_dish_loop()

    chef.speak_response(
        f"Great! Let's start cooking {chef.dish}. Ask me any question about it."
    )

    # ---------------------------
    # Continuous question-answer loop
    # ---------------------------
    while True:
        question = chef.listen("👂 Listening for your cooking question...")
        if not question:
            continue

        q_lower = question.lower()

        # Exit conditions
        if any(word in q_lower for word in ["quit", "exit", "stop", "bye"]):
            chef.speak_response("Thank you for cooking with me today. Happy cooking!")
            break
        if "thank" in q_lower:
            chef.speak_response("You're welcome! Happy cooking!")
            break

        # Generate answer
        chef.ask(question)

        # # Prompt for next question
        # chef.speak_response("Have any other questions?")
    #quit
    pygame.mixer.quit()


if __name__ == "__main__":
    start_voice_chat()
