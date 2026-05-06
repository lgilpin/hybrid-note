from openai import OpenAI
from dotenv import load_dotenv
import base64
import os
import io
import fitz  # PyMuPDF

# ------------------------
# Setup
# ------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=api_key)


# ------------------------
# Helpers
# ------------------------
def encode_image_file(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def encode_pil_image(img):
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def render_pdf_page_to_base64(page):
    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    img_bytes = pix.tobytes("jpeg")
    return base64.b64encode(img_bytes).decode("utf-8")


# ------------------------
# GPT Image Checks
# ------------------------
def classify_image_base64(image_base64):
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": """
Determine if this image should be treated as a scanned or image-based document for data extraction.

Return true if:
- it is a photo or scan of a document
- it contains handwritten notes
- it is a scanned printed page
- text appears embedded inside the image

Return false if:
- it is not a document
- it is clearly not notes or document content

If unsure, return true.

Answer ONLY with 'true' or 'false'.
"""
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        )

        result = response.output_text.strip().lower()
        return "true" in result

    except Exception as e:
        print("Error during image classification:", e)
        return True


def page_has_visual_annotations(image_base64):
    try:
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": """
Look at this PDF page image.

Return true if the page contains handwritten annotations, pen marks, drawings, added notes, circles, arrows, highlights, or markups on top of the document.

Return false if it is just a clean digital document with normal typed text and no visible annotations.

Answer ONLY with 'true' or 'false'.
"""
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{image_base64}"
                        }
                    ]
                }
            ]
        )

        result = response.output_text.strip().lower()
        return "true" in result

    except Exception as e:
        print("Error during annotation check:", e)
        return True


# ------------------------
# PDF Routing Logic
# ------------------------
def pdf_should_use_ocr(file_path, min_chars=100, max_pages=3):
    """
    Returns:
    True  -> OCR / vision pipeline
    False -> normal text extraction pipeline
    """

    try:
        doc = fitz.open(file_path)

        text = ""
        pages_to_check = min(max_pages, len(doc))

        for i in range(pages_to_check):
            page = doc[i]

            # 1. Check real PDF annotations
            annotations = list(page.annots() or [])
            if len(annotations) > 0:
                return True

            # 2. Collect selectable text
            text += page.get_text().strip()

        # 3. If no selectable text, likely scanned
        if len(text) < min_chars:
            return True

        # 4. It has selectable text, but may still have flattened handwriting/marks
        for i in range(pages_to_check):
            page = doc[i]
            image_base64 = render_pdf_page_to_base64(page)

            if page_has_visual_annotations(image_base64):
                return True

        # 5. Clean digital PDF
        return False

    except Exception as e:
        print("PDF check failed:", e)
        return True


# ------------------------
# Main Router
# ------------------------
def is_scanned_or_image_based_document(file_path):
    """
    Returns:
    True  -> OCR / vision extraction pipeline
    False -> normal text extraction pipeline
    """

    file_path_lower = file_path.lower()

    try:
        if file_path_lower.endswith((".jpg", ".jpeg", ".png", ".webp")):
            image_base64 = encode_image_file(file_path)
            return classify_image_base64(image_base64)

        elif file_path_lower.endswith(".pdf"):
            return pdf_should_use_ocr(file_path)

        else:
            print("Unsupported file type")
            return False

    except Exception as e:
        print("Error:", e)
        return True


# ------------------------
# Testing
# ------------------------
if __name__ == "__main__":
    print("Handwritten image:", is_scanned_or_image_based_document("notes_1.jpeg"))
    print("Digital PDF:", is_scanned_or_image_based_document("ShripadMangavalli_AI_ML.pdf"))
    print("Digital PDF with annotations:", is_scanned_or_image_based_document("AI_Panel_UC_Open_Oerview.pdf.pdf"))
    print("Digital PDF with images", is_scanned_or_image_based_document("Homework_2.pdf"))