# import os
# import tempfile
# import time
# from pathlib import Path
# import re

# # import pygame
# # import speech_recognition as sr
# from dotenv import load_dotenv
# from elevenlabs_functions import speak_text_to_stream
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_chroma import Chroma
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
# from openai import OpenAI
# from langchain.schema import HumanMessage, SystemMessage, AIMessage

# from models import ChatMessage, ChatRole
# from sqlmodel import select


# class MasterChefAssistant:
#     def __init__(self):
#         # self.recognizer = sr.Recognizer()
#         self.microphone = None
#         self.client = None
#         self.vector_db = None
#         self.retriever = None
#         self.llm = None
#         self.user_type = None
#         self.dish = None
#         self.recipe_list = {}
#         self.audio_cache = {}   # TTS caching
#         self.answer_cache = {}  # LLM caching

#     # ----------------------------- #
#     # INIT & SETUP
#     # ----------------------------- #
#     def initialize(self):
#         print("\n🍳 Master Chef Professional Voice Assistant\n" + "="*60)
#         try:
#             # self.microphone = sr.Microphone()
#             # pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=1024)

#             self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)
#             recipe_dir = Path("Recipe")
#             if recipe_dir.exists():
#                 for f in recipe_dir.glob("*.pdf"):
#                     self.recipe_list[f.stem.lower()] = f
#                 self.init_chain_for_all_recipe()

#             # with self.microphone as source:
#             #     print("🎤 Calibrating microphone (3s)...")
#             #     self.recognizer.adjust_for_ambient_noise(source, duration=3)
#             #     self.recognizer.dynamic_energy_threshold = True

#             print("✅ All systems ready!")
#             return True
#         except Exception as e:
#             print(f"❌ Init failed: {e}")
#             return False

#     # ----------------------------- #
#     # INIT LANGCHAIN / AI
#     # ----------------------------- #
#     def init_chain_for_all_recipe(self):
#         load_dotenv()
#         self.vector_db = Chroma(persist_directory="./chroma_rosendale", embedding_function=OpenAIEmbeddings())
#         for path in self.recipe_list.values():
#             loader = PyPDFLoader(str(path))
#             docs = loader.load()
#             for doc in docs:
#                 doc.metadata['source'] = str(path).split('\\')[-1]
#             splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
#             split_docs = splitter.split_documents(docs)
#             self.vector_db.add_documents(split_docs)
#         print('=====✅Initialized Successfully✅==')


#     def init_chain(self, selected_pdf):
#         load_dotenv()
#         self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#         loader = PyPDFLoader(str(selected_pdf))
#         docs = loader.load()

#         splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
#         split_docs = splitter.split_documents(docs)

#         vectordb = Chroma.from_documents(
#             split_docs, OpenAIEmbeddings(), persist_directory="./chroma_rosendale"
#         )
#         self.retriever = vectordb.as_retriever(search_kwargs={"k": 1})

#         self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

#     # ----------------------------- #
#     # TTS & LISTEN HELPERS
#     # ----------------------------- #
#     def preprocess_pronunciation(self, text: str) -> str:
#         corrections = {
#             "filets": "fill-AY",
#             "Filets": "Fill-AY",
#             "plancha": "plahn-cha",
#             "Plancha": "Plahn-cha",
#         }
#         for wrong, correct in corrections.items():
#             text = text.replace(wrong, correct)
#         return text

#     def speak_response(self, text):
#         try:
#             pass
#             # text = self.preprocess_pronunciation(text)
#             # if text in self.audio_cache:
#             #     audio_bytes = self.audio_cache[text]
#             # else:
#             #     audio_stream = speak_text_to_stream(text)
#             #     audio_bytes = audio_stream.getvalue()
#             #     self.audio_cache[text] = audio_bytes

#             # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
#             #     tmp.write(audio_bytes)
#             #     tmp_path = tmp.name

#             # pygame.mixer.music.load(tmp_path)
#             # pygame.mixer.music.play()
#             # while pygame.mixer.music.get_busy():
#             #     time.sleep(0.05)

#             # pygame.mixer.music.stop()
#             # pygame.mixer.music.unload()
#             # os.remove(tmp_path)
#         except Exception as e:
#             print(f"❌ Speech error: {e}")

#     def listen(self, prompt="Speak now..."):
#         pass
#         # print(prompt)
#         # for _ in range(3):
#         #     try:
#         #         with self.microphone as source:
#         #             audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=20)
#         #         return self.recognizer.recognize_google(audio, language="en-US")
#         #     except sr.WaitTimeoutError:
#         #         print("⏳ Timeout: no speech detected.")
#         #     except sr.UnknownValueError:
#         #         print("🤔 Could not understand. Please repeat.")
#         #     except Exception as e:
#         #         print(f"❌ Listen error: {e}")
#         # return None

#     # ----------------------------- #
#     # DYNAMIC RECIPE MATCHING
#     # ----------------------------- #
#     def set_dish(self, dish_name: str):
#         dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
#         dish_words = dish_clean.split()

#         print(dish_words, self.recipe_list)
#         match_count = {}

#         for name in self.recipe_list:
#             recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
#             match_count[name] = sum(1 for word in dish_words if word in recipe_clean)

#             print('=====match_count=======',match_count)
#         if match_count is not None:
#             best_key = max(match_count, key=match_count.get)   # 'fish_principles'
#             # best_value = match_count[best_key] 
#             self.dish = best_key
#             print('\n====self.dish===',self.dish)
#             return True
#             # self.init_chain(self.recipe_list[name])
#         return False


#     def set_dish(self, dish_name: str):
#         dish_clean = re.sub(r'[_\-]', ' ', dish_name.lower()).strip()
#         dish_words = dish_clean.split()

#         for name in self.recipe_list:
#             recipe_clean = re.sub(r'[_\-]', ' ', name.lower()).strip()
#             if all(word in recipe_clean for word in dish_words):
#                 self.dish = recipe_clean
#                 self.init_chain(self.recipe_list[name])
#                 return True
#         return False
    
#     def ask_dish_loop(self):
#         while True:
#             self.speak_response("What are we cooking today?")
#             answer = self.listen()
#             if not answer:
#                 self.speak_response("I couldn't hear you. Please try again.")
#                 continue
#             if self.set_dish(answer):
#                 return True
#             self.speak_response("I didn’t catch that recipe. Please say it again.")

#     # ----------------------------- #
#     # SET USER TYPE
#     # ----------------------------- #
#     def set_user_type(self, user_type: str):
#         if user_type.lower() in ["professional", "pro", "chef"]:
#             self.user_type = "Professional Chef"
#         else:
#             self.user_type = "Home Cook"

#     # ----------------------------- #
#     # AI RESPONSE (Updated for LangChain v0.1x)
#     # ----------------------------- #
#     def mentor_answer(self, user_id, question: str, session = None):
#         if question in self.answer_cache:
#             return self.answer_cache[question]

#         print('==Queation==',question)
#         try:
#             # Getting retriever from Recipe file
#             # file_name = str(self.recipe_list[ self.dish.replace(' ', '_') ]).replace('\\','/') if self.dish is not None else ''
#             # filter_dict = {"source": {"$in": [file_name]}}

#             # print(file_name)

#             retriever = self.vector_db.as_retriever() #(search_kwargs={"k": 5, "filter": filter_dict})

#             # Use get_relevant_documents instead of invoke
#             ctx_blocks = retriever.invoke(question)
#             if not ctx_blocks:
#                 return "This question cannot be answered from the provided Rosendale Method principles."

#             print('=====ctx_blocks===',len(ctx_blocks))

#             context = "\n".join(d.page_content for d in ctx_blocks[:3])[:1800]

#             # if self.user_type == "Professional Chef" or True:
#             # prompt = f"""
#             #     You are Chef Rosendale AI—a certified Master Chef and mentor for professional chefs.

#             #     Converse normally and warmly about culinary topics, industry trends, or kitchen operations using concise, professional language. Address the user’s questions directly, offer clear guidance, and maintain a confident, approachable tone. If a practical cooking or recipe question is asked (“How do I make...?”, “Guide me through this dish...”, “Step-by-step for...”, “How to sous vide...”, etc.):

#             #     - Shift to line-check briefing style for the answer.
#             #     - Speak in the first person as Chef Rosendale, using active, confident, and direct sentences.
#             #     - Avoid storytelling or filler outside of recipe responses.

#             #     When providing a recipe or step-by-step cooking guidance:
#             #     - Use only the context and facts supplied.
#             #     - Use advanced culinary terminology and avoid consumer language.
#             #     - Do not add headers, label sections, or repeat yourself.
#             #     - No external knowledge or narrative fluff.

#             #     For the detailed recipe, structure your response as follows: [ Never mention "Not specified" in any section, skip section if no valid information in present ]
#             #      Principle — state the main cooking goal and control points.
#             #      Method — stepwise technique, citing temperatures, timings, and key cues.
#             #      Tip — give a chef’s risk/mitigation or coordination insight.
#             #      Ingredients — specify critical details or variances from the provided content only.
#             #      Nutrients — offer a brief nutrition note if present, else state "Not specified."
#             #      Service — give instructions for plating, sauce ratio, pass timing, or finishing.

#             #     if asked specific recipe then use the following context, if valid:
#             #     {context}

#             #     If information is missing, briefly state a professional assumption and proceed. Otherwise, maintain a friendly, conversational style—inspire confidence, encourage inquiry, and foster a collaborative kitchen environment. Conduct the interaction conversationally, but deliver recipes and kitchen guidance with strict clarity, brevity, and line-ready detail.
#             # """
#             prompt = f"""
#                             YOU ARE ROZENDALE AI — a warm, conversational culinary guide inspired by Chef Rich Rosendale, CMC.

#             You speak like a real chef talking with a student, colleague, or guest — confident, calm, and approachable. Your job is to help people cook better, understand solid technique, solve real kitchen problems, and support anyone learning through Rosendale Online or the Rosendale Collective.
#             You do not dump long lists or read knowledge files word-for-word.
#             You simply draw from them naturally when helpful.
#             Background & Influences
#             When someone asks about your training, keep it simple and conversational. You can explain that your career was shaped by:
#             Training under Certified Master Chefs early on, which built your foundational discipline
#             Time in Michelin-starred kitchens during competitions and stages
#             Influences from world-class chefs you learned from during the Bocuse d’Or years
#             Global travel, classical apprenticeship, and modern technique development
#             If it feels right in context, you may casually mention chefs like:
#             Peter Timmins, CMC • Hartmut Handke, CMC • Laurence McFadden, CMC
#             Thomas Keller • Daniel Boulud • Grant Achatz
#             …but only when it feels natural — never as a long résumé list.
#             Culinary Style & Specialties
#             You have a wide range of experience, but you’re especially known for:
#             Sous Vide and Precision Cooking
#             American BBQ, but through a modern, technique-forward lens
#             Modern American cuisine
#             Classical European influences with modernist tools
#             Global cuisines picked up through travel, competition, and international training
#             When discussing food, you speak from real hands-on experience — blending classical fundamentals with modern innovation.
#             Tone & Style
#             You speak naturally:
#             No robotic phrasing
#             No heavy formatting
#             No long lists unless requested
#             Helpful, steady, and encouraging
#             Professional but friendly
#             Always grounded in real-world operations
#             Mission
#             Help people cook better, think better, and work smarter in the kitchen — whether they are professionals, students, or learning through Rosendale Online.
#             Boundaries
#             You never share sensitive personal information.
#             You keep things positive, safe, and professional.
#             You don’t exaggerate or overstate achievements.
#             You avoid sounding like a résumé.
#             You focus on genuine teaching.
#             👉 Example of How This Will Sound
#             User: “What’s your cooking style?”
#             Rosendale AI:
#             “I’m classically trained, but I’ve always loved blending technique with innovation. I do a lot of sous vide work, and I’ve spent years refining modern American BBQ. I also pull from global flavors I picked up through travel and competition training. My approach is really about precision, balance, and letting the food tell a story.”
#             Natural. Human. Confident. No lists, no robotic tone.
#                 """

#                 # f"""
#                 #     You are Chef Rosendale AI — a culinary mentor for PROFESSIONAL CHEFS.
#                 #     The student is cooking **{self.dish}**.

#                 #     CONTEXT:
#                 #     {context}

#                 #     QUESTION: {question}

#                 #     - Use precise culinary terminology, advanced techniques, and professional detail.
#                 #     - Structure with: Principle, Method, Tip, Ingredients, Nutrients.
#                 #     - Max 6 sentences.
#                 #     - Tone: confident, technical, authoritative.
#                 # """
#             # else:
#             #     prompt = f"""
#             #         You are Chef Rosendale AI — a culinary mentor for HOME COOKS.
#             #         The student is cooking **{self.dish}**.

#             #         CONTEXT:
#             #         {context}

#             #         QUESTION: {question}

#             #         - Use simple, approachable language with minimal steps.
#             #         - Structure with: Principle, Method, Tip, Ingredients, Nutrients.
#             #         - Max 6 sentences.
#             #         - Tone: warm, encouraging, beginner-friendly.
#             #     """

#             if not hasattr(self, 'llm'):
#                 self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

#             # Getting User Chat History
#             history = self.get_chat_history(user_id, session)
#             system_prompt = [ SystemMessage(content=prompt) ]
#             conversation_data = self.to_langchain_messages(history)
#             msg_now = [ HumanMessage(content=question) ]
#             # print('==conversation_data==',conversation_data)

#             # Updated LLM call
#             response = self.llm.invoke(system_prompt + conversation_data + msg_now).content.strip()

#             self.answer_cache[question] = response
#             return response

#         except Exception as e:
#             print(f"❌ mentor_answer error: {e}")
#             return "This question cannot be answered from the provided Rosendale Method principles."

#     def get_chat_history(self, user_id, session):
#         if session is None: return []
#         stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc())
#         if user_id is not None:
#             stmt = stmt.where(ChatMessage.user_id == user_id)
#         return session.exec(stmt).all()
    
#     # history: list[ChatMessage]
#     def to_langchain_messages(self, history):
#         conversation_data = []
#         for m in history[-20:]:  # keep last 20 turns
#             role = getattr(m, "type", None)
#             text = getattr(m, "text", "")
#             # normalize role to string
#             if hasattr(role, "value"):
#                 role = role.value
#             if role in ("user", "human"):
#                 conversation_data.append(HumanMessage(content=text))
#             else:
#                 # treat anything else as model/assistant
#                 conversation_data.append(AIMessage(content=text))
#         return conversation_data

#     def ask(self, question: str, speak: bool = True):
#         answer = self.mentor_answer('', question)
#         print("\n📋 CHEF'S RESPONSE:\n")
#         print(answer)
#         print("-"*60)
#         if speak:
#             self.speak_response(answer)
#         return answer


# # ----------------------------- #
# # INTERACTIVE MODE
# # ----------------------------- #
# def start_voice_chat():
#     chef = MasterChefAssistant()
#     if not chef.initialize():
#         return

#     chef.speak_response(
#         "Hello! I'm your Master Chef voice assistant. "
#         "I can guide you step-by-step through any recipe or cooking technique today."
#     )

#     # Set user type
#     chef.speak_response("Are you a professional chef, or a home cook?")
#     user_type = chef.listen()
#     if not user_type:
#         return
#     chef.set_user_type(user_type)

#     # Ask dish only once
#     chef.ask_dish_loop()

#     chef.speak_response(
#         f"Great! Let's start cooking {chef.dish}. Ask me any question about it."
#     )

#     # ---------------------------
#     # Continuous question-answer loop
#     # ---------------------------
#     while True:
#         question = chef.listen("👂 Listening for your cooking question...")
#         if not question:
#             continue

#         q_lower = question.lower()

#         # Exit conditions
#         if any(word in q_lower for word in ["quit", "exit", "stop", "bye"]):
#             chef.speak_response("Thank you for cooking with me today. Happy cooking!")
#             break
#         if "thank" in q_lower:
#             chef.speak_response("You're welcome! Happy cooking!")
#             break

#         # Generate answer
#         chef.ask(question)

#         # Prompt for next question
#         chef.speak_response("Have any other questions?")
#     #quit
#     pygame.mixer.quit()


# if __name__ == "__main__":
#     start_voice_chat()












import os
import tempfile
import time
from pathlib import Path
import re

from dotenv import load_dotenv
# from elevenlabs_functions import speak_text_to_stream   # optional TTS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

from models import ChatMessage, ChatRole   # keep if you use DB chat history
from sqlmodel import select

# ---------- Local uploaded file path (for reference) ----------
# Developer note: this is the local path to the uploaded system prompt file.
# If you need to convert to a URL for external calls, do that in your runtime environment.
UPLOADED_SYSTEM_PROMPT_PATH = "/mnt/data/FILE 1- system prompt.docx"


class MasterChefAssistant:
    def __init__(self):
        # voice/recognition placeholders (disabled by default)
        self.microphone = None
        self.client = None

        # RAG pieces
        self.vector_db = None
        self.retriever = None
        self.llm = None

        # state
        self.user_type = None
        self.dish = None
        self.recipe_list = {}   # dict: {stem_lower: Path}
        self.audio_cache = {}   # optional TTS cache
        self.answer_cache = {}  # LLM answer cache

    # ----------------------------- #
    # INIT & SETUP
    # ----------------------------- #
    def initialize(self):
        print("\n🍳 Master Chef Professional Voice Assistant\n" + "=" * 60)
        try:
            load_dotenv()
            # initialize LLM client (LangChain wrapper). Fine to reassign later.
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

            print(f"ℹ️ Uploaded system-prompt doc path (local): {UPLOADED_SYSTEM_PROMPT_PATH}")

            # scan Recipe directory for PDFs
            recipe_dir = Path("Recipe")
            if recipe_dir.exists():
                for f in recipe_dir.glob("*.pdf"):
                    self.recipe_list[f.stem.lower()] = f
                print(f"\n📁 PDFs detected in Recipe/: {len(self.recipe_list)}")
                for name, path in self.recipe_list.items():
                    print(f"   → {name}: {path}")
                # initialize vector store + chunk + index
                self.init_chain_for_all_recipe()
            else:
                print("⚠️ No Recipe folder found at ./Recipe. Skipping PDF load.")

            print("\n✅ Initialization complete. All systems ready!\n")
            return True
        except Exception as e:
            print(f"❌ Init failed: {e}")
            return False

    # ----------------------------- #
    # BATCH LOAD / CHUNK / INDEX ALL PDFs
    # ----------------------------- #
    def init_chain_for_all_recipe(self):
        print("\n📥 Initializing Vector Store & Loading PDFs...")
        load_dotenv()

        # instantiate Chroma vector DB (persisted directory)
        self.vector_db = Chroma(
            persist_directory="./chroma_rosendale",
            embedding_function=OpenAIEmbeddings(),
        )

        total_pages = 0
        total_chunks = 0
        loaded_files = 0

        for path in self.recipe_list.values():
            try:
                print(f"\n📄 Loading PDF: {path}")
                loader = PyPDFLoader(str(path))
                docs = loader.load()
                pages_count = len(docs)
                print(f"   → Pages loaded: {pages_count}")
                total_pages += pages_count
                loaded_files += 1

                # attach metadata and chunk
                for doc in docs:
                    doc.metadata["source"] = path.name

                splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
                split_docs = splitter.split_documents(docs)
                chunks_count = len(split_docs)
                total_chunks += chunks_count
                print(f"   → Chunks created: {chunks_count}")

                # add to vector DB
                self.vector_db.add_documents(split_docs)
                print("   → Added to Chroma successfully.")
            except Exception as e:
                print(f"   ❌ Failed loading {path}: {e}")

        print("\n==============================")
        print(f"📊 TOTAL PDF FILES PROCESSED: {loaded_files}")
        print(f"📚 TOTAL PAGES LOADED: {total_pages}")
        print(f"🧩 TOTAL CHUNKS STORED: {total_chunks}")
        print("==============================\n")

        print("✅ Vector Database Initialized Successfully!")

    # ----------------------------- #
    # Build per-PDF retriever if user selects a specific pdf/dish
    # ----------------------------- #
    def init_chain(self, selected_pdf: Path):
        """
        Build a local vectordb for a single selected pdf and create a retriever.
        (Useful for focusing on one recipe/document.)
        """
        print(f"\n🔧 Building focused retriever for: {selected_pdf}")
        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        loader = PyPDFLoader(str(selected_pdf))
        docs = loader.load()
        print(f"   → Pages loaded for selected file: {len(docs)}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=80)
        split_docs = splitter.split_documents(docs)
        print(f"   → Chunks created for selected file: {len(split_docs)}")

        vectordb = Chroma.from_documents(
            split_docs,
            OpenAIEmbeddings(),
            persist_directory="./chroma_rosendale",
        )
        self.retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        print("   → Focused retriever ready.")

        # ensure llm present
        if not self.llm:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

    # ----------------------------- #
    # TTS & LISTEN HELPERS (disabled stubs)
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

    def speak_response(self, text: str):
        # TTS is intentionally disabled in this runtime. Hook your TTS here.
        # Example (commented): audio_stream = speak_text_to_stream(self.preprocess_pronunciation(text))
        pass

    def listen(self, prompt="Speak now..."):
        # Speech recognition disabled for this environment.
        # Replace with your speech-to-text invocation if desired.
        return None

    # ----------------------------- #
    # DYNAMIC RECIPE MATCHING
    # ----------------------------- #
    def set_dish(self, dish_name: str):
        """
        Try to find a recipe PDF by matching words from dish_name to the recipe filenames.
        If found, sets self.dish and initializes focused retriever on that file.
        """
        dish_clean = re.sub(r"[_\-]", " ", dish_name.lower()).strip()
        dish_words = dish_clean.split()

        for name, path in self.recipe_list.items():
            recipe_clean = re.sub(r"[_\-]", " ", name.lower()).strip()
            # require all words present for a match
            if all(word in recipe_clean for word in dish_words):
                self.dish = name
                print(f"✅ Dish matched: {name} -> {path}")
                self.init_chain(path)
                return True
        print("⚠️ No matching recipe file found for:", dish_name)
        return False

    # ----------------------------- #
    # CHAT HISTORY HELPERS (DB-backed optional)
    # ----------------------------- #
    def get_chat_history(self, user_id, session):
        if session is None:
            return []
        stmt = select(ChatMessage).order_by(ChatMessage.created_at.desc())
        if user_id is not None:
            stmt = stmt.where(ChatMessage.user_id == user_id)
        return session.exec(stmt).all()

    def to_langchain_messages(self, history):
        """
        Convert stored ChatMessage objects into LangChain message objects.
        Uses last 20 turns to maintain context.
        """
        conversation_data = []
        for m in history[-20:]:
            role = getattr(m, "type", None)
            text = getattr(m, "text", "")
            if hasattr(role, "value"):
                role = role.value
            if role in ("user", "human"):
                conversation_data.append(HumanMessage(content=text))
            else:
                conversation_data.append(AIMessage(content=text))
        return conversation_data

    # ----------------------------- #
    # RAG: Retrieve + Generate (mentor_answer)
    # ----------------------------- #
    def mentor_answer(self, user_id, question: str, session=None):
        """
        Retrieve relevant document chunks and generate an answer using the Rosendale AI persona.
        """
        # return cached answer if available
        if question in self.answer_cache:
            print("🔁 Returning cached answer")
            return self.answer_cache[question]

        print("\n==QUESTION RECEIVED==")
        print(question)

        try:
            # Use vector_db retriever if available; fall back to focused retriever if set
            if self.vector_db is not None:
                retriever = self.vector_db.as_retriever(search_kwargs={"k": 3})
            elif self.retriever is not None:
                retriever = self.retriever
            else:
                print("⚠️ No retriever available. Please initialize vector DB first.")
                return "No indexed documents available."

            # retrieve candidate context blocks
            # Use .invoke(question) on some wrappers; if unavailable, use get_relevant_documents
            try:
                ctx_blocks = retriever.invoke(question)
            except Exception:
                ctx_blocks = retriever.get_relevant_documents(question)

            if not ctx_blocks:
                print("🔍 No context blocks retrieved.")
                return "This question cannot be answered from the provided Rosendale Method principles."

            print(f"🔍 Retrieved {len(ctx_blocks)} context blocks")
            # show brief preview for debugging
            for i, d in enumerate(ctx_blocks[:3], start=1):
                preview = d.page_content[:200].replace("\n", " ")
                print(f"   → Context[{i}] (source={d.metadata.get('source','unknown')}): {preview}...")

            # assemble context (limit to avoid overly long system message)
            context = "\n".join(d.page_content for d in ctx_blocks[:5])[:3000]

            # Build system prompt using the refined Rosendale style
            system_prompt = [
                SystemMessage(
                    content=(
                    "You are Rosendale AI — a warm, conversational culinary guide inspired by Chef Rich Rosendale, CMC.\n\n"
                "Speak like a real chef talking with a student or colleague: confident, calm, approachable, and grounded in hands-on kitchen experience.\n\n"
                "Purpose: help people cook better, understand solid technique, solve real kitchen problems, and support learners in Rosendale Online or the Rosendale Collective.\n\n"
                "Do not dump long lists or repeat documents. Draw from the provided context naturally, the way an experienced chef would. Never format responses as numbered steps, bullet points, or bold headings. No markdown lists. Explain things in flowing sentences, as if you were talking through the technique beside someone at the stove.\n\n"
                "BACKGROUND & INFLUENCE (use only when relevant): early training under Certified Master Chefs; Michelin-level stages; Bocuse d'Or experience; global travel & apprenticeship.\n\n"
                "CULINARY STYLE: Sous vide and precision cooking; modern, technique-forward American BBQ; modern American cuisine; classical European fundamentals supported by modernist tools.\n\n"
                "TONE: Natural, conversational, no robotic phrasing, no heavy formatting, no lists unless the user specifically asks. Helpful, steady, encouraging, professional.\n\n"
                "BOUNDARIES: Never share personal or sensitive info. Avoid résumé dumps and exaggeration. Focus on practical teaching, safety, and real kitchen logic.\n\n"
                "EXAMPLE: 'I’m classically trained, but I’ve always enjoyed blending technique with innovation. I lean heavily into sous vide...'\n\n"
                "CONTEXT FROM DOCUMENTS:\n"
                f"{context}\n\n"
                "When producing a response: stay conversational, concise, and line-ready when helpful. Avoid markdown lists. Explain technique like you're coaching someone live in the kitchen. Safety and clear reasoning always come first."

                    )
                )
            ]

            # conversation history (optional)
            history = self.get_chat_history(user_id, session) if session is not None else []
            conversation_data = self.to_langchain_messages(history) if history else []

            # current user query
            msg_now = [HumanMessage(content=question)]

            # ensure LLM exists
            if not self.llm:
                self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, max_tokens=400)

            # invoke LLM with system prompt + history + current question
            response_obj = self.llm.invoke(system_prompt + conversation_data + msg_now)
            response_text = response_obj.content.strip()

            # cache and return
            self.answer_cache[question] = response_text
            print("\n📋 GENERATED RESPONSE:\n")
            print(response_text)
            print("-" * 60)
            return response_text

        except Exception as e:
            print(f"❌ mentor_answer error: {e}")
            return "This question cannot be answered from the provided Rosendale Method principles."

    # ----------------------------- #
    # Simple ask wrapper
    # ----------------------------- #
    def ask(self, question: str, speak: bool = False):
        answer = self.mentor_answer("", question)
        if speak:
            self.speak_response(answer)
        return answer


# ----------------------------- #
# SIMPLE INTERACTIVE (CLI) MODE
# ----------------------------- #
def start_cli():
    chef = MasterChefAssistant()
    if not chef.initialize():
        return

    print("Type a question (or 'quit' to exit).")
    while True:
        question = input("\nAsk your question: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "stop", "bye"):
            print("Thank you. Happy cooking!")
            break
        # optional: allow selecting dish by prefix: "dish: salmon"
        if question.lower().startswith("dish:"):
            dish_name = question.split(":", 1)[1].strip()
            chef.set_dish(dish_name)
            continue

        chef.ask(question)


if __name__ == "__main__":
    start_cli()
