import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from fpdf import FPDF

# Load environment variables
load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")
genai.configure(api_key=google_api_key)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    if not text:
        raise ValueError("No text could be extracted from the PDF files.")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to create conversational chain for summarization
def get_summarization_chain():
    prompt_template = """
    Summarize the following legal document in a simple and understandable manner.
    Ensure that the summary captures the key points and legal implications clearly.

    Document:\n {context}\n

    Summary:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to summarize and translate the document
def summarize_and_translate(pdf_docs, target_language):
    try:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

        docs = new_db.similarity_search("Summarize this document")

        chain = get_summarization_chain()
        response = chain({"input_documents": docs, "question": "Summarize this document"}, return_only_outputs=True)
        summary = response["output_text"]

        # Translate the summary
        translation_prompt = f"Translate the following text to {target_language}. Keep the legal terminology accurate:\n{summary}"
        translation_model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=google_api_key)
        translated_summary = translation_model.invoke(translation_prompt).content

        return summary, translated_summary
    except Exception as e:
        raise gr.Error(f"An error occurred during processing: {str(e)}")

# Function to create a PDF from the summary
def create_pdf(summary, filename="summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary)
    pdf.output(filename)
    return filename

# Function to create a PDF from the translated summary
def create_translated_pdf(translated_summary, filename="translated_summary.pdf"):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, translated_summary)
    pdf.output(filename)
    return filename

# Gradio interface
def gradio_interface(pdf_docs, target_language):
    try:
        summary, translated_summary = summarize_and_translate(pdf_docs, target_language)
        pdf_filename = create_pdf(summary)
        translated_pdf = create_translated_pdf(translated_summary)
        return summary, translated_summary, pdf_filename, translated_pdf
    except Exception as e:
        return str(e), "", None, None

# Create the Gradio interface
iface = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.File(label="Upload PDF Files", file_count="multiple", file_types=[".pdf"]),
        gr.Dropdown(
            label="Target Language", 
            choices=["Tamil", "Telugu", "Hindi", "Kannada", "Malayalam"],
            value="Hindi"
        )
    ],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Translated Summary"),
        gr.File(label="Download Summary PDF"),
        gr.File(label="Download Translated Summary PDF")
    ],
    title="Legal Document Summarizer",
    description="Upload legal documents to get a summarized and translated version."
)

if __name__ == "__main__":
    try:
        iface.launch(server_name="0.0.0.0", server_port=7860, share=True)
    except Exception as e:
        print(f"Error launching Gradio interface: {str(e)}")
        print("Trying alternative launch method...")
        iface.launch(share=True, inbrowser=True)