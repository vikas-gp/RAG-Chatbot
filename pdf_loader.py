import re
import pdfplumber

def load_pdf(file_path):
    text = ""
    
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            extracted = page.extract_text()
            if extracted:
                text += extracted + '\n'
    
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text