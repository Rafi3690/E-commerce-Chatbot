from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.vectorstores import FAISS
from enum import Enum, auto
import os

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib import colors

# Add these imports at the top
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY

import os
from dotenv import load_dotenv
load_dotenv()


class ChatState(Enum):
    GREETING = auto()
    AGENCY_TYPE = auto()
    COMPANY_ANALYSIS = auto()
    MAIN_TASKS = auto()
    PAST_EXPERIENCES = auto()
    COLLABORATION_NEEDS = auto()
    FINAL_ANALYSIS = auto()
    DOCUMENT_CREATION = auto()

class MarketingAgencyChatbot:
    def __init__(self):
        os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv("HF_TOKEN")

        self.llm = ChatGroq(temperature=0.7, model_name="mixtral-8x7b-32768")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        self.memory = ConversationBufferMemory(return_messages=True)
        self.knowledge_base = self._create_knowledge_base()
        self.wikipedia = WikipediaAPIWrapper()
        self.current_state = ChatState.GREETING
        self.requirements = {}

    def _create_knowledge_base(self):
        agency_profiles = [
            "Advertising agency: Specializes in brand strategy, creative campaigns, and traditional media",
            "Digital agency: Focuses on web development, SEO, and social media marketing",
            "Media agency: Expertise in media planning, buying, and performance analytics"
        ]
        return FAISS.from_texts(agency_profiles, self.embeddings)

    def _get_chain(self, system_prompt):
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        return prompt | self.llm

    def _handle_greeting(self):
        response = "Welcome! I will assist you in creating a structured interview guide for agency selection.\n"
        response += "Let's start with understanding your needs. What type of agency are you looking for? "
        self.current_state = ChatState.AGENCY_TYPE
        return response

    def _handle_agency_type(self, user_input):
        docs = self.knowledge_base.similarity_search(user_input)
        context = "\n".join([d.page_content for d in docs])
        
        system_prompt = """Analyze the user's agency type needs and match with knowledge base. 
        Provide 2-3 key considerations for this agency type."""
        
        memory_vars = self.memory.load_memory_variables({})
        response = self._get_chain(system_prompt).invoke({
            "input": f"User needs: {user_input}\nContext: {context}",
            "history": memory_vars["history"]
        }).content
        
        self.memory.save_context({"input": user_input}, {"output": response})
        self.requirements["agency_type"] = user_input
        self.current_state = ChatState.COMPANY_ANALYSIS
        return f"{response}\n\nNow, for which company are we creating the guide? "

    def _handle_company_analysis(self, company_name):
        wiki_research = self.wikipedia.run(f"{company_name} marketing strategy")
        
        system_prompt = f"""Analyze company's main drivers from this Wikipedia data: 
        {wiki_research}. Identify top 3 marketing success factors in bullet points."""
        
        memory_vars = self.memory.load_memory_variables({})
        response = self._get_chain(system_prompt).invoke({
            "input": company_name,
            "history": memory_vars["history"]
        }).content
        
        self.memory.save_context({"input": company_name}, {"output": response})
        self.requirements["company_analysis"] = response
        self.current_state = ChatState.MAIN_TASKS
        return f"Based on my research, here are key success factors:\n{response}\n\nWhat are the three main tasks the agency should handle? "

    def _handle_main_tasks(self, user_input):
        self.memory.save_context({"input": user_input}, {"output": "Recorded main tasks"})
        self.requirements["main_tasks"] = user_input
        self.current_state = ChatState.PAST_EXPERIENCES
        return "Got it. Now, which two experiences with previous agencies had the strongest impact on your business? "

    def _handle_past_experiences(self, user_input):
        system_prompt = "Analyze these past agency experiences and extract key lessons:"
        memory_vars = self.memory.load_memory_variables({})
        
        response = self._get_chain(system_prompt).invoke({
            "input": user_input,
            "history": memory_vars["history"]
        }).content
        
        self.memory.save_context({"input": user_input}, {"output": response})
        self.requirements["past_experiences"] = response
        self.current_state = ChatState.COLLABORATION_NEEDS
        return f"Key lessons identified:\n{response}\n\nWhat are your top 3 collaboration needs? "

    def _handle_collaboration_needs(self, user_input):
        self.memory.save_context({"input": user_input}, {"output": "Recorded collaboration needs"})
        self.requirements["collaboration_needs"] = user_input
        self.current_state = ChatState.FINAL_ANALYSIS
        return self._generate_final_analysis()
    

    def _generate_final_analysis(self):
        # Format requirements into a single input string
        requirements_summary = "\n".join([
            f"- {key}: {value}" 
            for key, value in self.requirements.items()
            if key != "final_analysis"
        ])
        
        system_prompt = """Synthesize these requirements into 3-6 key criteria:
        {input}"""
        
        memory_vars = self.memory.load_memory_variables({})
        
        response = self._get_chain(system_prompt).invoke({
            "input": f"User Requirements:\n{requirements_summary}",
            "history": memory_vars["history"]
        }).content
        
        self.requirements["final_analysis"] = response
        self.current_state = ChatState.DOCUMENT_CREATION
        return f"Here's my proposed evaluation criteria:\n{response}\n\nGenerating your document now..."


    def process_input(self, user_input):
        if self.current_state == ChatState.GREETING:
            return self._handle_greeting()
        elif self.current_state == ChatState.AGENCY_TYPE:
            return self._handle_agency_type(user_input)
        elif self.current_state == ChatState.COMPANY_ANALYSIS:
            return self._handle_company_analysis(user_input)
        elif self.current_state == ChatState.MAIN_TASKS:
            return self._handle_main_tasks(user_input)
        elif self.current_state == ChatState.PAST_EXPERIENCES:
            return self._handle_past_experiences(user_input)
        elif self.current_state == ChatState.COLLABORATION_NEEDS:
            return self._handle_collaboration_needs(user_input)
        elif self.current_state == ChatState.FINAL_ANALYSIS:
            return self._generate_final_analysis()
        else:
            return "Document generation complete. Check your email!"
        

    def generate_pdf(self, filename="interview_guide.pdf"):
        doc = SimpleDocTemplate(filename, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Custom style that preserves newlines
        preserved_style = ParagraphStyle(
            name='Preserved',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=10,
            leading=12,
            spaceAfter=6,
        )
        
        def format_text(text):
            """Convert newlines to HTML breaks and preserve bullet points"""
            return text.replace('\n', '<br/>') if text else ''

        story = []
        
        # Title
        story.append(Paragraph(
            "<b>Structured Interview Guide for Agency Selection</b>", 
            styles["Title"]
        ))
        story.append(Spacer(1, 24))
        
        # Sections with preserved formatting
        sections = [
            ("Agency Type", 'agency_type'),
            ("Company Analysis", 'company_analysis'),
            ("Main Tasks", 'main_tasks'),
            ("Key Lessons from Past Experiences", 'past_experiences'),
            ("Collaboration Needs", 'collaboration_needs'),
            ("Evaluation Criteria", 'final_analysis')
        ]
        
        for title, key in sections:
            story.append(Paragraph(f"<b>{title}:</b>", styles["Heading2"]))
            content = format_text(self.requirements.get(key, ''))
            story.append(Paragraph(content, preserved_style))
            story.append(Spacer(1, 12))
        
        doc.build(story)
        return filename
  

    # Update the chat_loop method in the MarketingAgencyChatbot class
    def chat_loop(self):
        print("Chatbot: ", end="")
        response = self._handle_greeting()
        print(response)
        
        while self.current_state != ChatState.DOCUMENT_CREATION:
            user_input = input("User: ")
            print("Chatbot: ", end="")
            response = self.process_input(user_input)
            print(response)
        
        # Generate PDF
        filename = self.generate_pdf()
        print(f"\n[PDF document generated successfully: {filename}]")


# Start chatbot
bot = MarketingAgencyChatbot()
bot.chat_loop()