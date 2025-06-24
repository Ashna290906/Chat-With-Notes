import fitz
from io import BytesIO
import pandas as pd
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
    elif filetype in ["xlsx", "xls"]:
        # Read all sheets from the Excel file
        excel_data = pd.ExcelFile(file)
        text_parts = []
        
        for sheet_name in excel_data.sheet_names:
            df = pd.read_excel(excel_data, sheet_name=sheet_name)
            # Convert dataframe to string and clean up
            text_parts.append(f"--- Sheet: {sheet_name} ---")
            text_parts.append(df.to_string())
            
        return "\n".join(text_parts)
    else:
        raise ValueError(f"Unsupported file type: {filetype}")

def split_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    return splitter.split_text(text)
