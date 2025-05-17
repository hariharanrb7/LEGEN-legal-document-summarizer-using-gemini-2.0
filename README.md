# ⚖️ LEGEN - Legal Document Summarizer using Gemini 2.0

LEGEN is an AI-powered application designed to **summarize**, **translate**, and **narrate** lengthy legal documents using **Google Gemini 2.0**, **LangChain**, and **FAISS**. Built with **Streamlit**, it empowers legal professionals and general users to quickly digest complex legal content in multiple formats.

---

## 🚀 Features

- 📄 **PDF Legal Document Summarization**
- 🌐 **Multi-language Translation** (supports 10+ languages)
- 🔊 **Text-to-Speech** Narration (generate audio from summaries)
- 📘 **PDF Export** (summarized content in PDF format)
- 💬 **Conversational QA** from documents (RAG using Gemini + FAISS)
- 🧠 **Multimodal AI Integration** via Gemini 2.0 Pro

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini Pro 2.0 (via LangChain)
- **RAG**: FAISS Vector Store
- **PDF Handling**: PyPDF2, FPDF
- **Audio**: gTTS (Google Text-to-Speech)
- **Translation**: Gemini multilingual prompt engineering

---

## 🧠 How It Works
1. User uploads a PDF document.

2. Text is extracted using PyPDF2.

3. Text is chunked and embedded using LangChain + FAISS.

4. Gemini Pro 2.0 is queried via LangChain for:

  - Summarization

  - Translation

  - Q&A using Retrieval-Augmented Generation (RAG)

5. Summary is:

  - Shown in UI

  - Translated (optional)

  - Exported as PDF (optional)

  - Narrated via audio (optional)

---

---

## 🌍 Supported Languages for Translation
- English

- Tamil

- Hindi

- Malayalam

- Telugu

- Kannada

- Marathi

---

## 👨‍💻 Author

Hariharan R.B

Generative AI Enthusiast

