// ------------------- File Upload -------------------
let uploadedFile = null;

function handleFileUpload() {
  const fileInput = document.getElementById("pdfUpload");
  const fileName = document.getElementById("fileName");
  const viewBtn = document.getElementById("viewBtn");

  if (fileInput.files.length > 0) {
    uploadedFile = fileInput.files[0];
    fileName.textContent = uploadedFile.name;
    viewBtn.disabled = false;
  } else {
    fileName.textContent = "No file selected";
    viewBtn.disabled = true;
    uploadedFile = null;
  }
}

function viewPDF() {
  if (!uploadedFile) return;
  const fileURL = URL.createObjectURL(uploadedFile);
  window.open(fileURL, "_blank");
}

// ------------------- Summarization -------------------
async function summarizePDF() {
  if (!uploadedFile) {
    alert("Please select a PDF first!");
    return;
  }

  const formData = new FormData();
  formData.append("file", uploadedFile);

  const summaryBox = document.getElementById("summaryBox");
  summaryBox.classList.remove("collapsed");
  summaryBox.classList.add("expanded");
  summaryBox.innerHTML = "⏳ Uploading & summarizing... please wait.";

  try {
    // --- Upload PDF ---
    const uploadRes = await fetch("/upload", { method: "POST", body: formData });
    const uploadData = await uploadRes.json();

    if (!uploadRes.ok || uploadData.error) {
      throw new Error(uploadData.error || "File upload failed");
    }

    // --- Summarize PDF ---
    const summarizeRes = await fetch("/summarize", { method: "POST" });
    const summarizeData = await summarizeRes.json();

    if (!summarizeRes.ok || summarizeData.error) {
      throw new Error(summarizeData.error || "Summarization failed");
    }

    summaryBox.innerHTML = renderMarkdown(summarizeData.summary || "⚠️ No summary generated.");

  } catch (error) {
    console.error(error);
    summaryBox.innerHTML = `❌ Error during upload/summarization: ${error.message}`;
  }
}

// ------------------- Q/A Chat -------------------
async function askQuestion() {
  const input = document.getElementById("userQuestion");
  const chatBox = document.getElementById("chatBox");
  const question = input.value.trim();
  if (!question) return;

  // Display user message
  const userMessage = document.createElement("p");
  userMessage.innerHTML = `<strong>You:</strong> ${question}`;
  chatBox.appendChild(userMessage);

  // Show bot typing indicator
  const botTyping = document.createElement("p");
  botTyping.innerHTML = `<strong>Bot:</strong> ⏳ typing...`;
  chatBox.appendChild(botTyping);
  chatBox.scrollTop = chatBox.scrollHeight;

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    botTyping.innerHTML = `<strong>Bot:</strong> ${renderMarkdown(data.answer || "⚠️ No answer returned.")}`;
  } catch (error) {
    botTyping.innerHTML = `<strong>Bot:</strong> ❌ Error getting response: ${error.message}`;
    console.error(error);
  }

  chatBox.scrollTop = chatBox.scrollHeight;
  input.value = "";
}

// ------------------- Simple Markdown Renderer -------------------
function renderMarkdown(text) {
  let html = text.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
  const lines = html.split("\n");
  let listItems = [];
  let otherLines = [];

  lines.forEach(line => {
    line = line.trim();
    if (/^\d+\./.test(line)) { 
      listItems.push(`<li>${line}</li>`);
    } else if (line.startsWith("• ") || line.startsWith("- ")) {
      listItems.push(`<li>${line.slice(2)}</li>`);
    } else {
      otherLines.push(`<p>${line}</p>`);
    }
  });

  let listHTML = listItems.length ? `<ul>${listItems.join("")}</ul>` : "";
  return listHTML + otherLines.join("");
}

// ------------------- Reset Site -------------------
function resetPage() {
  fetch("/reset", { method: "POST" })
    .then(res => {
      if (res.ok) window.location.reload();
      else alert("Failed to reset site. Try again.");
    })
    .catch(err => {
      console.error(err);
      alert("Error resetting site.");
    });
}
