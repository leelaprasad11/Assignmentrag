import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Set up your API key securely
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

class BookExpertBot:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.vector_store = None
        self.retrieval_chain = None
        self.chat_history = []  # Maintains conversational memory
        
        # Initialize pipeline components
        self._prepare_knowledge_base()
        self._build_rag_chain()

    def _prepare_knowledge_base(self):
        """Loads the book/document, splits it into chunks, and creates vector embeddings."""
        print(f"Parsing document: {self.file_path}...")
        
        # Support both PDF and TXT extensions
        if self.file_path.endswith('.pdf'):
            loader = PyPDFLoader(self.file_path)
        else:
            loader = TextLoader(self.file_path, encoding='utf-8')
            
        documents = loader.load()

        # Chunking strategy optimized for books (remembers context across paragraphs)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} text chunks.")

        # Vector store creation
        embeddings = OpenAIEmbeddings()
        self.vector_store = FAISS.from_documents(chunks, embeddings)
        print("Vector store successfully built.")

    def _build_rag_chain(self):
        """Constructs the conversational RAG chain with memory."""
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 4})

        # Contextualize question: Reformulates query if it refers to past chat history
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it."
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # Core Answer Prompt for the "Book Expert"
        system_prompt = (
            "You are 'BookExpert', an expert AI assistant specialized in analyzing documents.\n"
            "Answer the question using strictly the provided context below. If the answer is "
            "not present in the context, politely state that the information is not available "
            "in the text.\n\n"
            "Context:\n{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

        # Combine into full retrieval chain
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        self.retrieval_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def ask(self, query: str):
        """Executes a user query, tracks history, and extracts source references."""
        if not self.retrieval_chain:
            return "System not initialized."

        response = self.retrieval_chain.invoke({
            "input": query,
            "chat_history": self.chat_history
        })

        # Save to chat history
        self.chat_history.append(HumanMessage(content=query))
        self.chat_history.append(AIMessage(content=response["answer"]))

        # Extract sources for accountability
        sources = []
        for doc in response.get("context", []):
            page = doc.metadata.get("page", "N/A")
            sources.append(f"Chunk Preview: '{doc.page_content[:60]}...' (Page/Source: {page})")

        return {
            "answer": response["answer"],
            "sources": sources
        }

# --- Quick Test Execution ---
if __name__ == "__main__":
    # 1. Create a dummy file if testing locally
    sample_book = "book_reference.txt"
    if not os.path.exists(sample_book):
        with open(sample_book, "w", encoding="utf-8") as f:
            f.write("Chapter 1: The Foundations of AI. Artificial Intelligence began as a formal academic discipline in 1956. "
                    "The term RAG stands for Retrieval-Augmented Generation, which optimizes LLM outputs using external knowledge.")

    # 2. Run Bot
    bot = BookExpertBot(sample_book)
    
    # Query 1
    query_1 = "When did AI become a formal academic discipline?"
    res_1 = bot.ask(query_1)
    print(f"\nQ: {query_1}\nA: {res_1['answer']}")
    
    # Query 2 (Testing conversational memory context: 'it')
    query_2 = "What does the term RAG stand for mentioned in it?"
    res_2 = bot.ask(query_2)
    print(f"\nQ: {query_2}\nA: {res_2['answer']}")
          
