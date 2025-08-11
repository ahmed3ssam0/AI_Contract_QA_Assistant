import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer
from io import BytesIO, StringIO
import docx2txt
import PyPDF2
import torch
import tempfile
import os

# ========== OCR DEPENDENCIES ==========
try:
    from PIL import Image
    import pytesseract

    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    st.warning("OCR capabilities disabled. Install pytesseract and Pillow for image text extraction.")

# ========== CONFIG ==========
st.set_page_config(page_title="Legal-BERT QA", layout="centered")

# Load model and tokenizer with fallback
model_path = r"finetuned-legal-bert"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # FIX: Use low_cpu_mem_usage=False to prevent meta tensor issues
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, low_cpu_mem_usage=False)

    # Check for meta tensors and initialize weights if needed
    if any(param.device == torch.device('meta') for param in model.parameters()):
        # Initialize weights properly
        model = model.to_empty(device='cpu')
        state_dict = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if torch.cuda.is_available():
        # Use safe half-precision conversion
        model = model.half() if model.dtype != torch.float16 else model
except Exception as e:
    st.warning(f"Error loading fine-tuned model: {e}. Falling back to pre-trained model.")
    model_path = "deepset/bert-base-uncased-squad2"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForQuestionAnswering.from_pretrained(model_path, low_cpu_mem_usage=False)
    model = model.to(device)
    if torch.cuda.is_available():
        model = model.half() if model.dtype != torch.float16 else model

st.title("⚖ Legal Document Question Answering")
st.write("Upload documents or images and ask multiple questions without re-uploading.")

# ========== FILE UPLOAD ==========
uploaded_file = st.file_uploader("Upload a .txt, .docx, .pdf, or image file",
                                 type=["txt", "docx", "pdf", "png", "jpg", "jpeg"])


def extract_text(file):
    """Extract text from various file types including images using OCR"""
    file_type = file.name.split('.')[-1].lower()

    if file_type == "txt":
        return StringIO(file.getvalue().decode("utf-8")).read()

    elif file_type == "docx":
        return docx2txt.process(file)

    elif file_type == "pdf":
        reader = PyPDF2.PdfReader(file)
        return "\n".join(page.extract_text() for page in reader.pages if page.extract_text())

    elif file_type in ["png", "jpg", "jpeg"] and OCR_AVAILABLE:
        try:
            image = Image.open(file)
            text = pytesseract.image_to_string(image)
            return text
        except Exception as e:
            st.error(f"OCR Error: {str(e)}")
            return ""

    elif file_type in ["png", "jpg", "jpeg"] and not OCR_AVAILABLE:
        st.error("OCR not available. Install pytesseract and Pillow for image text extraction.")
        return ""

    return ""


if uploaded_file and "file_text" not in st.session_state:
    try:
        st.session_state.file_text = extract_text(uploaded_file)
        if not st.session_state.file_text.strip():
            st.warning("No text could be extracted from the file.")
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")

if "file_text" in st.session_state and st.session_state.file_text:
    st.text_area("📘 Extracted Text from File:",
                 value=st.session_state.file_text,
                 height=300,
                 help="This is the text extracted from your uploaded document")


# ========== TOKEN-BASED CHUNKING ==========
def chunk_text(text, tokenizer, max_tokens=300):
    """Split text into chunks based on token count"""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    for i in range(0, len(tokens), max_tokens):
        yield tokenizer.decode(tokens[i:i + max_tokens])


# ========== QA FUNCTION ==========
def answer_question(question, context, model, tokenizer):
    try:
        max_length = model.config.max_position_embeddings  # Usually 512

        # Tokenize with length checks
        inputs = tokenizer(
            question,
            context,
            truncation="only_second",  # Only truncate context
            max_length=max_length,
            return_tensors="pt",
            padding="max_length"
        )

        # Verify input length
        input_length = inputs["input_ids"].shape[1]
        if input_length > max_length:
            return "⚠️ Context too long after tokenization", float('-inf')

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        start_logits = outputs.start_logits[0].cpu().numpy()
        end_logits = outputs.end_logits[0].cpu().numpy()

        # Find best answer span
        max_score = float('-inf')
        start_idx = end_idx = 0

        # Consider only valid positions (ignore special tokens)
        valid_start_end = list(range(0, len(start_logits)))

        for i in valid_start_end:
            # Only consider j within a reasonable range from i
            for j in range(i, min(i + 30, len(end_logits))):
                if j not in valid_start_end:
                    continue
                score = start_logits[i] + end_logits[j]
                if score > max_score:
                    start_idx = i
                    end_idx = j
                    max_score = score

        # Extract valid answer
        if max_score == float('-inf') or start_idx == 0:
            return "No answer found in document", max_score

        answer_tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        return answer.strip(), max_score

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            return "⚠️ GPU memory error. Try shorter context.", float('-inf')
        return f"⚠️ Error: {str(e)}", float('-inf')


# ========== QUESTION INPUT ==========
st.markdown("---")
user_question = st.text_area("🧠 Ask your question about the uploaded document:",
                             placeholder="What is the effective date of this agreement?",
                             height=100)

if st.button("Get Answer", type="primary"):
    if "file_text" not in st.session_state:
        st.warning("Please upload a document first.")
    elif not st.session_state.file_text.strip():
        st.warning("The uploaded document contains no text.")
    elif not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Analyzing document..."):
            best_answer = "No valid answer found"
            best_score = float('-inf')
            chunks = list(chunk_text(st.session_state.file_text, tokenizer, max_tokens=300))

            progress_bar = st.progress(0)
            for i, chunk in enumerate(chunks):
                progress_bar.progress((i + 1) / len(chunks))
                answer, score = answer_question(user_question, chunk, model, tokenizer)
                if score > best_score:
                    best_score = score
                    best_answer = answer

            progress_bar.empty()

        st.success("✅ Answer:")
        st.subheader(best_answer)
        st.caption(f"Confidence score: {best_score:.2f}")