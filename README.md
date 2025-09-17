📄 ResearchXpert
================

**Illuminate Research Papers with Interactive Conversational AI**

ResearchXpert is a web application that allows users to upload research papers (PDFs) and interact with them through AI-powered summarization and Q&A. The app extracts text from uploaded PDFs, generates embeddings, and uses a large language model (LLM) to provide detailed summaries and answer user queries.

🚀 Features
-----------

### 📑 PDF Summarization

*   Upload research papers in PDF format.
    
*   Generate concise, high-quality summaries of the paper.
    
*   Supports long documents via chunking and embeddings.
    

### 💬 Chat n Seek (Q&A)

*   Ask questions about the uploaded paper.
    
*   AI provides contextual answers based on the content.
    
*   Supports interactive exploration of research papers.
    

### 🧰 Additional Features

*   View uploaded PDF directly in the browser.
    
*   Reset/Refresh button to clear all uploads and chat history.
    
*   Clean, modern UI with collapsible summary boxes and chat bubbles.
    

🎨 Frontend
-----------

*   HTML5 for structure
    
*   CSS3 with modern gradients, shadows, and hover effects
    
*   Vanilla JavaScript (ES6) for:
    
    *   File upload handling
        
    *   Summarization requests
        
    *   Q&A interaction
        
    *   Reset/Refresh functionality
        

🖥️ Backend
-----------

*   **Python 3.13**
    
*   **Flask** for routing and serving frontend
    
*   **PyMuPDF (fitz)** for PDF text extraction
    
*   **FAISS** for storing and searching text embeddings
    
*   **Groq LLM API** for summarization and Q&A
    

⚙️ Project Structure
--------------------

    smart-paper-explainer/
    │
    ├── app.py                  # Flask backend with routes
    ├── embeddings_utils.py      # Text embedding and FAISS index functions
    ├── qa_utils.py              # Groq API integration for summarization & Q/A
    ├── text_utils.py            # Text cleaning and chunking
    ├── static/
    │   ├── style.css            # Frontend CSS
    │   └── script.js            # Frontend JS
    ├── templates/
    │   └── index.html           # Frontend HTML
    ├── uploads/                 # Temporary storage for uploaded PDFs
    ├── requirements.txt         # Python dependencies
    └── README.md                # Project documentation

`

💻 Setup Instructions
---------------------

1.  **Clone the repository**
    
      git clone https://github.com/Shivanggaryaa/smart-paper-explainer.git  cd smart-paper-explainer   `

1.  **Create a virtual environment**
    
      python -m venv venv
    
      source venv/bin/activate           # Linux/Mac  
      venv\Scripts\activate              # Windows   `

1.  **Install dependencies**

      pip install -r requirements.txt   `

1.  **Configure environment variables**
    

*   Create a .env file in the root directory:
      GROQ_API_KEY=your_groq_api_key   `

1.  **Run the Flask app**

      python app.py   `


🔧 Usage
--------

1.  Upload a research paper (PDF) via the **Upload** section.
    
2.  Click **Summarize** to generate a concise summary.
    
3.  Ask questions in the **Chat n Seek** section.
    
4.  Use the **Reset/Refresh** button to clear all uploads and start fresh.
    

🛠️ Tech Stack
--------------

*   **Frontend:** HTML5, CSS3, JavaScript (ES6)
    
*   **Backend:** Python, Flask
    
*   **PDF Processing:** PyMuPDF
    
*   **Embeddings & Search:** FAISS
    
*   **LLM API:** Groq API (LLaMA 3.1)
    

📌 Future Improvements
----------------------

*   Add user authentication for saving sessions.
    
*   Persistent storage for uploaded PDFs and summaries.
    
*   Enhanced UI/UX for mobile responsiveness.
    
*   Better handling for very large PDFs.
    
*   Multiple LLM backends for summarization and Q&A.




🔖 Live Demo :   https://huggingface.co/spaces/ShivangArya/rshxpt
