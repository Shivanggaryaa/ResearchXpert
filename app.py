import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import shutil
import fitz  # PyMuPDF

# Global state (kept small)
chunks = []
index = None
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)

# ---------------------- PDF Extraction ----------------------
def extract_text_from_pdf(file_path: str) -> str:
    with fitz.open(file_path) as doc:
        return "\n".join(p.get_text("text") for p in doc)


# ---------------------- Routes ----------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_pdf():
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

        # Extract and process lazily
        from text_utils import clean_text, chunk_text
        from embeddings_utils import embed_texts, build_faiss_index

        text = clean_text(extract_text_from_pdf(filepath))
        chunks = chunk_text(text)

        # Build FAISS index
        vectors = embed_texts(chunks)
        index = build_faiss_index(vectors)

        return jsonify({"status": "success", "filename": filename})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/summarize", methods=["POST"])
def summarize_pdf():
    global chunks, index
    if not chunks or not index:
        return jsonify({"error": "No document uploaded yet"}), 400

    try:
        from qa_utils import answer_with_groq
        from scholar_utils import find_related_papers

        summary, _ = answer_with_groq("Summarize this paper.", chunks, index)
        related = find_related_papers(summary)

        return jsonify({"summary": summary, "related_papers": related})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask_question():
    global chunks, index
    if not chunks or not index:
        return jsonify({"error": "No document uploaded yet"}), 400

    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        from qa_utils import answer_with_groq
        answer, _ = answer_with_groq(question, chunks, index)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset_site():
    global chunks, index
    chunks, index = [], None

    for filename in os.listdir(UPLOAD_FOLDER):
        try:
            path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(path):
                os.unlink(path)
        except Exception as e:
            print(f"Delete failed: {e}")

    return "Reset successful", 200


# ---------------------- Main ----------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
