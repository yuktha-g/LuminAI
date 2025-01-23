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

2. Install Required Packages
   ```bash
   pip install -r requirements.txt
3. Environment Setup
*	Create a .env file in the root directory
*	Add the following environment variables:
    ```bash
    MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.1
    EMBEDDING_MODEL=hkunlp/instructor-xl

### Running the Application
* Start the streamlit app:
    ```bash
    streamlit chatbot_LuminAI.py
    
*	Open browser and navigate to: http://localhost:8501


## Usage Instructions
1.	Upload Documents:
   * Drag and drop files in "Document Library" 
   *	Supported formats: PDF, DOCX, PPTX, LaTeX
2.	Ask Questions:
   * Enter queries in text input field
   * LuminAI will analyze the uploaded documents and generate responses.
3. Download History:
   * Export chat history as .txt


## File Structure
Copy
LuminAI/
├── app.py                # Main Streamlit application
├── document_processor.py # Document processing logic
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── README.md             # Project documentation
├── htmlTemplates2.py     # UI templates
└── ...                   # Other supporting files


## Note
* Ensure that the uploaded documents contain relevant research papers with readable text. The effectiveness of the chatbot depends on the quality and relevance of the content provided.
* For better performance, it is recommended to provide clear and concise questions related to the content of the research papers.


## Conclusion

Lumin AI transforms the landscape of data analysis and decision-making by providing cutting-edge, intuitive tools for businesses and researchers. With its advanced machine learning algorithms and user-friendly interface, Lumin AI empowers users to uncover hidden patterns, automate complex processes, and generate actionable insights with ease. Whether for predictive analytics, trend identification, or real-time decision support, Lumin AI is the ultimate companion for anyone looking to leverage data to its fullest potential. Step into the future of intelligent data processing with Lumin AI today!

##
Please find the demo here

![Screenshot (296)](https://github.com/user-attachments/assets/a1907346-dc15-4cad-a209-f1a30c62172d)
![Screenshot (297)](https://github.com/user-attachments/assets/0e05b0fb-6061-4e68-b6bf-5bfc2511f9ab)

![Screenshot (298)](https://github.com/user-attachments/assets/8eef3450-4802-4946-b65b-15432bad3434)

![Screenshot (299)](https://github.com/user-attachments/assets/4e4d5a59-9493-4180-a86d-99703b84e411)

![Screenshot (300)](https://github.com/user-attachments/assets/3afb9ce2-4c0a-4b23-b091-3d85e7716026)

![Screenshot (301)](https://github.com/user-attachments/assets/2dd58c50-3b56-413c-896d-32617832db7e)


![Screenshot (302)](https://github.com/user-attachments/assets/24f26010-3cc7-4cf5-aa06-54a4b2ec4a8c)



