import os
import fitz  # PyMuPDF
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import shutil

# Import your custom utils
from text_utils import clean_text, chunk_text
from embeddings_utils import embed_texts, build_faiss_index
from qa_utils import answer_with_groq
from scholar_utils import find_related_papers

app = Flask(__name__)

# ---- Global state (demo only, for session) ----
chunks = []
index = None
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# ---------------------- PDF Extraction ----------------------
def extract_text_from_pdf(file_path: str) -> str:
    """Extract all text from a PDF file path."""
    with fitz.open(file_path) as doc:
        return "\n".join(p.get_text("text") for p in doc)


# ---------------------- Routes ----------------------
@app.route("/")
def home():
    """Serve frontend HTML (index.html)."""
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
    """Handle PDF upload, extract text, chunk, and embed."""
    global chunks, index

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Extract + clean + chunk
        full_text = extract_text_from_pdf(filepath)
        text = clean_text(full_text)
        chunks = chunk_text(text)

        # Embed + build FAISS index
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)

        return jsonify({"status": "success", "filepath": filepath, "filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    """Generate a summary and related papers."""
    global chunks, index

    if not chunks or not index:
        return jsonify({"error": "No document uploaded yet"}), 400

    try:
        # Enhanced summary
        summary, _ = answer_with_groq(
            "Summarize this paper.",
            chunks,
            index,
            keywords=["SPGNN", "EfficientNet", "R-Plot32"]
        )
        related_papers = find_related_papers(summary)
        return jsonify({"summary": summary, "related_papers": related_papers})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    """Answer a user’s question about the uploaded paper."""
    global chunks, index

    if not chunks or not index:
        return jsonify({"error": "No document uploaded yet"}), 400

    data = request.get_json()
    question = data.get("question", "")

    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        # Enhanced Q/A with structured, bullet-pointed response
        answer, _ = answer_with_groq(
            question,
            chunks,
            index,
            keywords=["SPGNN", "EfficientNet", "R-Plot32"]
        )
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    


@app.route("/reset", methods=["POST"])
def reset_site():
    """Clear uploads folder and reset server-side state."""
    global chunks, index

    # Clear in-memory variables
    chunks = []
    index = None

    # Delete all files in uploads folder
    upload_folder = "uploads"
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

    return "Reset successful", 200


# ---------------------- Main ----------------------
if __name__ == "__main__":
    app.run(debug=True)
