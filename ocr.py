from pypdf import PdfReader
import pytesseract
from pdf2image import convert_from_path
import tempfile
import os

def extract_text_from_pdf(pdf_file_path):
    text = ""
    pdf_reader = PdfReader(open(pdf_file_path, 'rb'))
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        try:
            text += page.extract_text()
        except:
            # If extract_text() fails, use OCR
            with tempfile.TemporaryDirectory() as path:
                images = convert_from_path(pdf_file_path, output_folder=path)
                for image in images:
                    text += pytesseract.image_to_string(image)
    return text