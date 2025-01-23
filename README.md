# LuminAI: Your Intelligent Research Companion

LuminAI is an AI-driven research assistant designed to enhance your document interaction experience. It seamlessly processes a variety of document formats, including PDFs, DOCX, PPTX, and LaTeX files, to provide accurate and contextually relevant answers to user queries. With LuminAI, navigating through research documents becomes efficient and intuitive.

---

## Features

### 1. Document Format Support
* Supports multiple formats:
  * **PDF**
  * **DOCX**
  * **PPTX**
  * **LaTeX**

### 2. Intelligent Document Processing
* Extracts and processes text efficiently from uploaded documents.
* Handles multiple files simultaneously for comprehensive analysis.

### 3. Natural Language Understanding
* Delivers accurate and context-aware responses to user queries.
* Includes source citations for generated answers.

### 4. Local AI Model Integration
* Utilizes open-source models such as **LLaMA** and **GPT-2** for text generation.
* Operates without reliance on paid APIs like OpenAI or Hugging Face Hub.

### 5. Interactive Web Interface
* Built on **Streamlit**, featuring:
  * A clean and intuitive UI.
  * Dynamic typewriter-style response animations.

### 6. Chat History Management
* Users can download their question-answer history as a `.txt` file for future reference.

---

## Tech Stack

* **Python 3.8+**
* **Streamlit**: Interactive web interface.
* **LangChain**: Advanced text processing.
* **PyPDF2, python-docx, pptx**: Text extraction from documents.
* **Transformers**: AI-driven NLP tasks.
* **dotenv**: For managing environment variables.

---

## Workflow Description

1. **File Upload**:
   * Users upload research documents in supported formats.
   * Documents are parsed, and text is chunked for efficient processing.

2. **Embedding and Query Processing**:
   * Text embeddings are generated using models like `hkunlp/instructor-xl`.
   * Queries are converted into dense vector representations for similarity matching.

3. **Ranked Retrieval and Language Model Integration**:
   * Relevant content is retrieved based on vector similarity.
   * Open-source LLMs, such as **Mistral-7B-Instruct**, are used for generating responses.

4. **Output Generation**:
   * Contextually accurate answers are delivered with source citations.

5. **Downloadable History**:
   * Users can save their chat history for future reference.

---

## Installation and Setup

### Prerequisites:
* Ensure the following are installed:
  * **Python 3.8+**
  * **pip** (Python package manager)

### Installation Steps:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/luminai.git
   cd luminai


