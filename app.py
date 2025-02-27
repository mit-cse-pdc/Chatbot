import streamlit as st  # Importing Streamlit for building the web application
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility  # Importing Milvus library for database operations
from sentence_transformers import SentenceTransformer  # Importing SentenceTransformer for generating embeddings
from PyPDF2 import PdfReader  # Importing PdfReader for reading PDF files
from langchain_community.llms import HuggingFaceHub  # Importing HuggingFaceHub for language model integration
from langchain.prompts import PromptTemplate  # Importing PromptTemplate for creating prompts for the language model
from langchain.schema.runnable import RunnableLambda  # Importing RunnableLambda for creating runnable functions
from docx import Document  # Importing Document for reading DOCX files
import os  # Importing os for environment variable access
from langchain_community.document_loaders import UnstructuredExcelLoader
import openai
from dotenv import load_dotenv
load_dotenv()  # Load API keys from .env

# Define connection parameters for Milvus database
MILVUS_HOST = "in03-ff4505225900ecd.serverless.gcp-us-west1.cloud.zilliz.com"  # Host address for Milvus server
MILVUS_PORT = "443"  # Port for Milvus server
MILVUS_USER = "db_ff4505225900ecd"  # Username for Milvus authentication
MILVUS_PASSWORD = "Dh4;{,Y6>ROOl{.."  # Password for Milvus authentication


def connect_to_milvus():
    """Connect to the Milvus database using the defined parameters."""
    try:
        # Disconnect any existing connection with the alias "default"
        if "default" in connections.list_connections():
            connections.disconnect(alias="default")

        # Connect to the Milvus database
        connections.connect(
            alias="default",
            host=MILVUS_HOST,  # Host address for the Milvus server
            port=MILVUS_PORT,  # Port for the Milvus server
            user=MILVUS_USER,  # Username for Milvus authentication
            password=MILVUS_PASSWORD,  # Password for Milvus authentication
            secure=True  # Use secure connection
        )
        print("✅ Successfully connected to Milvus!")
    except Exception as e:
        print(f"❌ Failed to connect to Milvus: {e}")
        raise SystemExit("Terminating script. Please check your credentials or Milvus server status.")


# ❌ Calling connect_to_milvus() before defining MILVUS_* variables
connect_to_milvus()  # This will fail if MILVUS_HOST is not defined before this call




# Define or Load Collection
collection_name = "document_embeddings"  # Name of the collection to store document embeddings


if collection_name not in utility.list_collections():
    # Define fields for the collection
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),  # Field for unique ID
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768),  # Field for storing embeddings
        FieldSchema(name="text_chunk", dtype=DataType.VARCHAR, max_length=2000),  # Field for storing text chunks
        FieldSchema(name="file_name", dtype=DataType.VARCHAR, max_length=255),  # Field for storing file names
    ]
    schema = CollectionSchema(fields, description="Embedding storage for document chunks")  # Define the schema for the collection
    collection = Collection(name=collection_name, schema=schema)  # Create the collection
    # Create index with COSINE similarity for efficient searching
    collection.create_index(
        field_name="embedding",
        index_params={"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 128}}
    )
else:
    collection = Collection(name=collection_name)  # Load the existing collection


def load_excel(file_path):
    """Load text data from an Excel file using LangChain's UnstructuredExcelLoader."""
    loader = UnstructuredExcelLoader(file_path)  # Load entire Excel file
    documents = loader.load()  # Loads Excel as a list of Document objects
    return documents  # Return extracted text

# Helper Functions
def load_pdf(file_path):
    """Load text from a PDF file and return a list of lowercased text pages."""
    reader = PdfReader(file_path)  # Create a PDF reader object
    return [page.extract_text().strip().lower() for page in reader.pages if page.extract_text()]  # Extract and return text


def load_docx(file_path):
    """Load text from a DOCX file and return a list of lowercased paragraphs."""
    doc = Document(file_path)  # Create a Document object
    return [p.text.strip().lower() for p in doc.paragraphs if p.text.strip()]  # Extract and return text


def chunk_text(text_list, chunk_size=500, overlap=50):
    """Chunk the text into smaller pieces with specified size and overlap."""
    chunks = []
    for text in text_list:
        for i in range(0, len(text), chunk_size - overlap):
            chunks.append(text[i:i + chunk_size])  # Append each chunk to the list
    return chunks  # Return the list of chunks


def store_embeddings(file_paths):
    """Store embeddings for the provided file paths in the Milvus collection."""
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the sentence transformer model
    collection.load()  # Ensure the collection is loaded
    for file in file_paths:
        file_name = file.split("/")[-1]  # Extract the file name from the path
        # Check if embeddings for the file already exist
        if collection.query(f'file_name == "{file_name}"', output_fields=["file_name"]):
            st.info(f"✅ Embeddings for '{file_name}' already exist. Skipping...")
            continue  # Skip to the next file if embeddings exist
        # Load text from the file and chunk it
        if file.endswith(".pdf"):
            chunks = load_pdf(file)
        elif file.endswith(".docx"):
            chunks = load_docx(file)
        elif file.endswith(".xlsx"):
            documents = load_excel(file)  # Load Excel file
            chunks = [doc.page_content for doc in documents]  # Extract text chunks
        else:
            st.error(f"Unsupported file format: {file}")
            return

        chunks = chunk_text(chunks)  # Chunk the text
        embeddings = model.encode(chunks, convert_to_tensor=False)  # Generate embeddings
        # Match the schema fields for insertion
        data = [embeddings.tolist(), chunks, [file_name] * len(chunks)]
        collection.insert(data)  # Insert data into the collection
        collection.flush()  # Flush the collection to ensure data is saved
        st.success(f"✅ Stored {len(chunks)} chunks from '{file_name}' in Milvus!")  #<diff>



def query_milvus(query, top_k=3, extra_context=3):
    """Query the Milvus collection for relevant information based on the user's query."""
    collection.load()  # Load the collection
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")  # Load the sentence transformer model
    query_embedding = model.encode([query.lower()], convert_to_tensor=False)  # Generate embedding for the query


    # 🛠 Otherwise, perform a normal search
    search_results = collection.search(
        data=query_embedding.tolist(),
        anns_field="embedding",
        param={"metric_type": "COSINE", "params": {"nprobe": 10}},
        limit=top_k,
        output_fields=["text_chunk", "id", "file_name"]
    )

    if not search_results[0]:
        return "No relevant information found.", None

    # Step 2️⃣: Extract the Top-K Most Relevant Chunks
    matched_chunks = [(res.entity.get("id"), res.entity.get("text_chunk")) for res in search_results[0]]

    # Step 3️⃣: Fetch Extra Surrounding Chunks for Better Context
    surrounding_chunks = []
    for chunk_id, _ in matched_chunks:
        surrounding = collection.query(
            expr=f"id >= {chunk_id - extra_context} and id <= {chunk_id + extra_context}",
            output_fields=["text_chunk", "id", "file_name"]
        )
        surrounding_chunks.extend(surrounding)

    # Step 4️⃣: Merge and Filter Duplicate Chunks
    unique_chunks = {chunk["id"]: chunk["text_chunk"] for chunk in surrounding_chunks}
    sorted_context = "\n\n".join(unique_chunks[key] for key in sorted(unique_chunks))

    # Step 5️⃣: Get all unique file names from the relevant chunks
    file_names = list(set(chunk["file_name"] for chunk in surrounding_chunks if "file_name" in chunk))


    # Step 6️⃣: Pass the Enhanced Context to the LLM
    answer = run_llm(query, sorted_context)

    return answer, file_names  # ✅ Now the correct source will be displayed



def run_llm(query, context):
    """Run OpenAI GPT-4o-mini model to generate an answer while keeping Hugging Face integration intact."""

    # Load OpenAI API key from .env file
    openai.api_key = os.getenv("OPENAI_API_KEY")

    prompt = f"""You are an AI assistant that must ONLY use the provided syllabus context.
    
    Some of the context may be irrelevant or not directly related to the question.

    ### Step 1️⃣: Filter the Context
    - Identify the most relevant part of the syllabus context for answering the question.
    - Ignore anything unrelated to the topic of the question.

    ### Step 2️⃣: Generate a Precise Answer
    - Answer the question strictly using the relevant syllabus content.
    - DO NOT add any extra knowledge.
    - DO NOT include references unless they are directly relevant.

    **Question:** {query}

    **Syllabus Context:**
    {context}

    **Step 1️⃣: Filtered Relevant Context:** (AI selects relevant part)
    **Step 2️⃣: Final Answer:**
    """

    try:
        # OpenAI client initialization (corrected for latest OpenAI SDK)
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,  # Keep the response deterministic
            max_tokens=512  # Prevent excessively long responses
        )

        return response.choices[0].message.content.strip()  # Extract the response

    except openai.AuthenticationError:
        return "❌ Invalid OpenAI API key. Please check your API key in the .env file."

    except openai.RateLimitError:
        return "⚠️ OpenAI API rate limit reached. Please wait or check your billing settings."

    except openai.OpenAIError as e:
        return f"❌ OpenAI API Error: {e}"

    except Exception as e:
        return f"⚠️ Unexpected error: {e}"



pdf_links = {
    #"CSE(AI&ML) (Paragraph Syllabus).pdf": "https://drive.google.com/uc?export=download&id=1o-pC6tvFrpy9fo-90HrVtB27QNM5zX-p",
    # "Faculty list with email id.pdf": "https://drive.google.com/uc?export=download&id=1orS4Yw79kW34k5hRufREdw5C4BrcaoZ9",
    # "Cabin No..pdf": "https://drive.google.com/uc?export=download&id=1uGhL5jpfRl3GEjFt0qAa2evcPc2C_XHq",
    # "Leave request form":"https://drive.google.com/uc?export=download&id=1lNOTgDvAxOfke4Q4M8zP4Opwn3DRSKGC",
    "Attendance process .pdf": "https://drive.google.com/file/d/1-CnXO3c3V__LXRPAc_dQPCB-0hNBhaL2/view?usp=sharing"
}



# Display the chatbot title
st.title("💬 Academic Chatbot")
#st.write("Upload PDF, DOCX, or XLSX files and ask questions!")

# Maintain chat history using session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# ✅ GPT-style chat history (show older messages first)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):  # Displays as chat bubbles
        st.markdown(message["content"])

# Create a chat input field at the bottom of the page
if query := st.chat_input("Ask a question..."):

    # Store and display the user query
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    # Retrieve answer from Milvus-based search
    with st.spinner("Retrieving context..."):
        response, file_names = query_milvus(query)

    # Extract only the final answer (removing unnecessary context)
    cleaned_response = response.split("Step 2️⃣: Final Answer:")[-1].strip()

    # Display response from chatbot
    with st.chat_message("assistant"):
        st.markdown(cleaned_response)

    # Store response in session state
    st.session_state.messages.append({"role": "assistant", "content": cleaned_response})

    # ✅ Ensure only the latest question's sources are displayed
    st.session_state.latest_sources = file_names if file_names else []  # Store sources only for this query

# Display only the sources for the latest question
if "latest_sources" in st.session_state and st.session_state.latest_sources:
    filtered_sources = [file for file in st.session_state.latest_sources if file in pdf_links]
    if filtered_sources:
        st.markdown("### 📄 Sources for the Latest Question:")
        for file in filtered_sources:
            st.markdown(f"- [📄 {file}]({pdf_links[file]})", unsafe_allow_html=True)

        