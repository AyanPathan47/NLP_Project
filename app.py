"""
app.py - Flask web application entry point.
Handles routing and connects the frontend to the summarizer backend.
"""

import os
import tempfile

from flask import Flask, render_template, request
from summarizer import summarize
from utils.pdf_reader import extract_text_from_pdf

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")


@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/summarizer", methods=["GET", "POST"])
def summarizer_page():
    """Show the summarizer form (GET) or process input and display results (POST)."""

    summary_result = None
    original_text = ""
    error_message = None
    selected_length = "medium"

    if request.method == "POST":
        selected_length = request.form.get("summary_length", "medium")
        uploaded_file = request.files.get("pdf_file")

        if uploaded_file and uploaded_file.filename != "":
            # PDF upload
            if not uploaded_file.filename.lower().endswith(".pdf"):
                error_message = "Please upload a valid PDF file."
            else:
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        uploaded_file.save(tmp.name)
                        tmp_path = tmp.name

                    original_text = extract_text_from_pdf(tmp_path)
                    os.unlink(tmp_path)  # clean up temp file

                    if not original_text or not original_text.strip():
                        error_message = (
                            "Could not extract any text from the PDF. "
                            "The file may contain only images."
                        )
                except Exception as e:
                    error_message = f"Error reading PDF: {str(e)}"
        else:
            # Plain text input
            original_text = request.form.get("input_text", "").strip()

        # Run summarizer if we have valid text
        if original_text and not error_message:
            if len(original_text.split()) < 30:
                error_message = (
                    "Please enter at least 30 words so the summarizer "
                    "has enough content to work with."
                )
            else:
                try:
                    summary_result = summarize(original_text, selected_length)
                except Exception as e:
                    error_message = f"Summarization error: {str(e)}"

        if not original_text and not error_message:
            error_message = "Please enter some text or upload a PDF file."

    return render_template(
        "index.html",
        summary_result=summary_result,
        original_text=original_text,
        error_message=error_message,
        selected_length=selected_length,
    )


if __name__ == "__main__":
    print("\nNotes Summarizer running at http://127.0.0.1:5000\n")
    app.run(debug=os.environ.get("FLASK_DEBUG", "false").lower() == "true")
