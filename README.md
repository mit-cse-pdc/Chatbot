# Academic Chatbot - Streamlit Application

## Overview
This is an AI-powered academic chatbot built using **Streamlit**, **Milvus**, **Sentence Transformers**, and **OpenAI GPT-4o-mini**. It allows users to upload **PDF, DOCX, and XLSX** documents and ask questions based on the embedded content. The chatbot retrieves relevant information from the stored documents and generates responses using OpenAI's language model.

## Features
- 📄 **Upload and process documents (PDF, DOCX, XLSX)**
- 🔍 **Retrieve context from document embeddings using Milvus**
- 💡 **Generate accurate answers using OpenAI GPT-4o-mini**
- 🗂️ **GPT-style chat UI with chat history persistence**
- 🔗 **Show sources related to the latest query**
- 🔧 **Uses environment variables to securely load OpenAI API keys**

## Tech Stack
- **Frontend**: Streamlit (for UI)
- **Embedding Model**: Sentence Transformers (`all-mpnet-base-v2`)
- **Vector Database**: Milvus
- **LLM**: OpenAI GPT-4o-mini
- **Document Parsing**: PyPDF2, python-docx, UnstructuredExcelLoader

## Installation & Setup
### 1️⃣ Prerequisites
Ensure you have **Python 3.8+** installed. Install required dependencies:
```sh
pip install -r requirements.txt
```

### 2️⃣ Set Up Environment Variables
Create a `.env` file in the root directory and add your **OpenAI API Key**:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 3️⃣ Run the Application
Start the Streamlit app:
```sh
streamlit run app.py
```

## How It Works
1. **Document Upload**: Users upload files which are processed and chunked.
2. **Embedding Storage**: Sentence embeddings are generated and stored in Milvus.
3. **User Query**: User inputs a question via the chat interface.
4. **Context Retrieval**: The most relevant text chunks are retrieved from Milvus.
5. **LLM Processing**: OpenAI GPT-4o-mini generates a response based on the retrieved context.
6. **Source Display**: Only the latest query's sources are shown.

## Folder Structure
```
📂 project-folder
 ┣ 📜 app.py              # Main Streamlit app
 ┣ 📜 requirements.txt    # Required dependencies
 ┣ 📜 .env                # Stores OpenAI API key
 ┣ 📜 README.md           # Documentation
```

## Known Issues & Debugging
### 🔴 **Sources Not Displaying Properly?**
- Ensure filenames match exactly with `pdf_links` keys.
- Debug by printing `file_names` before filtering:
  ```python
  st.write(f"Raw file names from query_milvus: {file_names}")
  ```

### 🔴 **Milvus Connection Issues?**
- Check if the credentials are correct in the `connect_to_milvus()` function.
- Restart the Milvus server if necessary.

### 🔴 **OpenAI API Errors?**
- Ensure `.env` contains a valid `OPENAI_API_KEY`.
- Check API limits on OpenAI’s platform.

## Future Enhancements
- ✅ Add support for additional file formats (CSV, TXT, etc.)
- ✅ Implement UI improvements for better user experience
- ✅ Improve query optimization and answer relevance

---



