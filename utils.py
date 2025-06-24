import fitz
from io import BytesIO
from docx import Document
from pptx import Presentation
from langchain.text_splitter import RecursiveCharacterTextSplitter

def extract_text(file, filetype):
    if filetype == "pdf":
        doc = fitz.open("pdf", file.read())
        return "\n".join(page.get_text() for page in doc)
    elif filetype == "docx":
        doc = Document(file)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    elif filetype == "pptx":
        prs = Presentation(file)
        return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))
    else:
        raise ValueError("Unsupported file type.")

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)
