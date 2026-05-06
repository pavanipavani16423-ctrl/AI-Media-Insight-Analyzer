📊 AI Media Insight Analyzer

AI-powered web application to analyze media transcripts (news, podcasts, YouTube videos) and generate insights such as summaries, sentiment analysis, topic extraction, and intelligent Q&A using RAG (Retrieval-Augmented Generation).

---

🚀 Features

- Upload or paste media transcripts
- Automatic summary generation
- Sentiment analysis (Positive / Negative / Neutral)
- Topic extraction
- RAG-based Question & Answer system
- Fast responses using Google Gemini 2.5 Flash

---

🛠️ Tech Stack

- Frontend: Streamlit
- Backend: Python
- LLM: Google Gemini 2.5 Flash
- Embeddings: Gemini Embeddings
- Framework: LangChain
- Vector Database: ChromaDB

---

📂 Project Structure

AI-Media-Insight-Analyzer/
│
├── app.py
├── requirements.txt
├── README.md
│
├── utils/
│   ├── embeddings.py
│   ├── rag_pipeline.py
│   └── analysis.py
│
└── data/
└── transcripts/

---

⚙️ Installation

1. Clone the repository
   git clone https://github.com/github.com/pavanipavani16423/AI-Media-Insight-Analyzer.git
   cd AI-Media-Insight-Analyzer

2. Create virtual environment
   python -m venv venv
   venv\Scripts\activate

3. Install dependencies
   pip install -r requirements.txt

---



---

▶️ Run the Application

streamlit run app.py

---

🧠 How It Works

1. User inputs transcript
2. Text is converted into embeddings
3. Stored in ChromaDB
4. Gemini generates:
   - Summary
   - Sentiment
   - Topics
5. RAG answers user queries

---

📈 Future Improvements

- YouTube video integration
- Multi-language support
- Real-time analysis
- Dashboard visualization

---

👩‍💻 Author

Pavani

---

⭐ Support

If you like this project, give it a ⭐ on GitHub!
