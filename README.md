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
* Runs on Local Model.
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
   * Semantic content retrieval
   * Response generation

4. **Output Generation**:
   * Contextually accurate answers are delivered with source citations.

5. **Downloadable History**:
   * Users can save their chat history for future reference.

---
##
Please find the demo here
![Screenshot (296)](https://github.com/user-attachments/assets/c1b89eb5-8baa-4d22-b354-15a7f5cd73a1)

![Screenshot (297)](https://github.com/user-attachments/assets/6e038aab-2cc4-458e-9eab-e393b668a567)

![Screenshot (298)](https://github.com/user-attachments/assets/3dbbda2a-f337-4ebb-9695-02dc19eeeea7)

![Screenshot (299)](https://github.com/user-attachments/assets/727ef849-d33f-4662-973b-5e26cac752a4)

![Screenshot (300)](https://github.com/user-attachments/assets/ba603e93-67c9-4faf-ad41-95ba5450e371)


![Screenshot (301)](https://github.com/user-attachments/assets/6e12c517-5f8e-4c34-ba1c-4f68f0f96df8)

![Screenshot (302)](https://github.com/user-attachments/assets/cb9511c1-4462-4012-804c-bd0af160bcda)

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





