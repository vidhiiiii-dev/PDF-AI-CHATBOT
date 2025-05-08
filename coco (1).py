import gradio as gr
import pytesseract
from PIL import Image
import pdfplumber
import requests
import os
from docx import Document
from sentence_transformers import SentenceTransformer
import faiss

HUGGINGFACE_API_TOKEN = "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"

embed_model = SentenceTransformer('all-MiniLM-L6-v2')

documents = []
vectors = []
index = faiss.IndexFlatL2(384)


def extract_text(file):
    try:
        ext = file.name.split(".")[-1].lower()
        if ext == "pdf":
            with pdfplumber.open(file.name) as pdf:
                return "\n".join([page.extract_text() or "" for page in pdf.pages])
        elif ext in ["png", "jpg", "jpeg"]:
            return pytesseract.image_to_string(Image.open(file.name))
        elif ext == "docx":
            doc = Document(file.name)
            return "\n".join([p.text for p in doc.paragraphs])
        else:
            return file.read().decode()
    except Exception as e:
        print(f"Error extracting text: {str(e)}")
        return f"Error extracting text: {str(e)}"

def process_doc(file):
    if file is None:
        return [["Error", "❌ No file uploaded. Please upload a file first."]]
    
    try:
        raw_text = extract_text(file)
        chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]
        global documents, vectors, index
        documents.extend(chunks)
        chunk_vecs = embed_model.encode(chunks)
        vectors.extend(chunk_vecs)
        index.add(chunk_vecs)
        return [["Upload Done", "✅ You can start asking now."]]
    except Exception as e:
        return [["Error", f"❌ An error occurred: {str(e)}"]]

#if working then use this model model="mistralai/Mistral-7B-Instruct-v0.1"
def query_huggingface(prompt, model="HuggingFaceH4/zephyr-7b-beta"):
    API_URL = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
    payload = {"inputs": prompt, "parameters": {"max_new_tokens": 256}}

    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        return f"⚠️ API Error: {response.status_code} - {response.text}"
    output = response.json()
    try:
        return output[0]["generated_text"].split("Answer:")[-1].strip()
    except:
        return output[0]["generated_text"]


def chat_with_doc(message, history):
    if len(vectors) == 0:
        return history + [[message, "❗ Upload a document first."]]

    q_vec = embed_model.encode([message])
    _, I = index.search(q_vec, k=3)
    context = "\n".join([documents[i] for i in I[0]])

    prompt = f"""You are a helpful assistant. Use the context to answer the user's question.

Context:
{context}

Question:
{message}

Answer:"""

    print("==== Prompt ====\n", prompt)
    response = query_huggingface(prompt)
    if not response:
        response = "No response. Try asking again."
    history.append([message, response])
    return history


with gr.Blocks(css="""
    .upload-btn, .send-btn {
        border-radius: 25px !important;
        background: radial-gradient(circle at center, #3F5EFB, #FC466B) !important;
        background-size: 200% 200% !important;
        animation: radial-gradient-animation 3s linear infinite !important;
        font-weight: bold !important;
        color: white !important;
        border: none !important;
        padding: 10px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
        transition: transform 0.3s !important;
    }
    
    .upload-btn:hover, .send-btn:hover {
        transform: translateY(-3px) !important;
        box-shadow: 0 6px 20px rgba(0,0,0,0.25) !important;
    }
    
    .upload-btn {
        height: 45px !important;
        min-width: 80px !important;
        margin-top: 25px !important;
    }
    
    .send-btn {
        height: 45px !important;
        min-width: 80px !important;
        margin-top: 25px !important;
    }
    
    .input-container, .upload-container {
        display: flex !important;
        align-items: center !important;
        gap: 10px !important;
    }
    
    .input-container > div:first-child, .upload-container > div:first-child {
        flex-grow: 1 !important;
    }
    
    @keyframes radial-gradient-animation {
        0% {
            background-position: 0% 0%;
            background-size: 100% 100%;
        }
        50% {
            background-position: 100% 100%;
            background-size: 200% 200%;
        }
        100% {
            background-position: 0% 0%;
            background-size: 100% 100%;
        }
    }
""") as demo:
    gr.Markdown("""
    <div style="text-align:left; margin: 20px 0;">
        <p style="
            font-size: 18px;
            padding: 10px;
            background: radial-gradient(circle at center, #3494E6, #EC6EAD);
            border-radius: 8px;
            color: white;
            font-weight: bold;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);">
            DocGPT- Your Document Assistant 
        </p>
    </div>
    """)
    
    with gr.Row(elem_classes=["upload-container"]):
        file_input = gr.File(label="📂 Upload PDF / DOCX / Image / TXT")
        upload_btn = gr.Button("UPLOAD", elem_classes=["upload-btn"])
    
    chatbot = gr.Chatbot()
    
    with gr.Row(elem_classes=["input-container"]):
        user_input = gr.Textbox(label="💬 Ask something", show_label=True)
        send_btn = gr.Button("Send", elem_classes=["send-btn"])

    upload_btn.click(process_doc, inputs=file_input, outputs=chatbot)
    send_btn.click(chat_with_doc, inputs=[user_input, chatbot], outputs=chatbot)
    

    gr.Markdown("""
    <div style="text-align:center; margin-top: 20px;">
        <p style="
            font-weight: bold;
            font-size: 16px;
            color: white,black;">
            Created by 
            <span style="
                background: radial-gradient(circle at center, #002BFF, #F70031);
                background-size: 200% 200%;
                animation: radial-gradient-animation 3s linear infinite;
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;">
                Ananya Singh, Vidhi Mitra
            </span>
        </p>
    </div>
    
    <style>
        @keyframes radial-gradient-animation {
            0% {
                background-position: 0% 0%;
                background-size: 100% 100%;
            }
            50% {
                background-position: 100% 100%;
                background-size: 200% 200%;
            }
            100% {
                background-position: 0% 0%;
                background-size: 100% 100%;
            }
        }
    </style>
    """)

demo.launch(share=True,pwa=True)