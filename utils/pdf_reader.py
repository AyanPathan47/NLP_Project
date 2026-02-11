"""
pdf_reader.py - Extract text from PDF files using pdfplumber.
"""

import pdfplumber


def extract_text_from_pdf(pdf_path):
    """Read all pages of a PDF and return combined text.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        All extracted text joined by newlines, or empty string
        if no text was found.
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)

    return "\n".join(all_text)
