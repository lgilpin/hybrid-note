#!/usr/bin/env python3
"""
PDF classifier: determines whether a PDF is a scanned document or
digitally generated, using a VLM-only pipeline with Claude Haiku.

Usage:
    python pdf_classifier.py <path/to/file.pdf>
    python pdf_classifier.py <path/to/folder/>
"""
# <<Imports>>
import argparse
import base64
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import anthropic
import fitz  # PyMuPDF

# <<Setup>>
MODEL = "claude-haiku-4-5"
DPI = 150  # Upping the page resolution for higher accuracy
DOCS_FOLDER = Path("docs")

SYSTEM_PROMPT = """You are a document forensics expert who determines whether a \
PDF page was produced by scanning a physical document or by digital software.

Examine the page image carefully looking for signs that point it to either direction. Here are some things to look for, but are not limited to using to determine the correct result:

SCANNED indicators
- Skewed or rotated text lines (even slightly)
- Off-white background with noise or grain.
- Shadow near page edges or spine
- Ink issues: bleeding, feathering, or uneven stroke width
- Soft or slightly blurry text edges
- Speckle noise or visible paper texture
- Inconsistent baseline alignment across lines
- Physical signs: fingerprints, smudges, staple marks, fold lines, etc.

DIGITAL indicators
- Perfectly uniform white or solid-color background
- Crisp, mathematically sharp text edges
- Perfectly horizontal baselines
- Uniform, consistent font rendering
- Absence of noise or grain
- Vector-quality graphics, borders, and charts
- Consistent spacing and typographic grid
- Clean, anti-aliased glyphs with no ink spread

Respond with ONLY a valid JSON object. No prose, no markdown fences:
{
  "classification": "scanned" | "digital",
  "confidence": <float 0.0–1.0>,
  "evidence": ["<short evidence phrase>", ...]
}"""

# <<Data Classes>>
@dataclass
class PageResult:
    page_number: int
    classification: str   # "scanned" | "digital"
    confidence: float
    evidence: list[str] = field(default_factory=list)


@dataclass
class DocumentResult:
    pdf_path: str
    total_pages: int
    overall_classification: str
    overall_confidence: float
    per_page: list[PageResult] = field(default_factory=list)


# <<Helpers>>

def render_page_to_base64(page: fitz.Page, dpi: int = DPI) -> str:
    """Render a PDF page to a base64-encoded PNG string."""
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
    return base64.standard_b64encode(pix.tobytes("png")).decode("utf-8")


def _build_system() -> list[dict]:
    """System prompt block with prompt-caching enabled."""
    return [
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral"},
        }
    ]


def _parse_model_response(text: str) -> dict:
    """Extract and parse the JSON payload from Claude's response."""
    text = text.strip()
    # Strip optional markdown code fences
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop first and last fence lines
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        # Drop optional language tag
        if inner and not inner[0].strip().startswith("{"):
            inner = inner[1:]
        text = "\n".join(inner).strip()
    return json.loads(text)


def classify_page(
    client: anthropic.Anthropic,
    image_b64: str,
    page_num: int,
    system: list[dict],
) -> PageResult:
    """Send one page image to Claude and return its classification."""
    response = client.messages.create(
        model=MODEL,
        max_tokens=512,
        system=system,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Classify this document page. "
                            "Reply with the JSON object only."
                        ),
                    },
                ],
            }
        ],
    )

    raw_text = next(b.text for b in response.content if b.type == "text")
    data = _parse_model_response(raw_text)

    return PageResult(
        page_number=page_num,
        classification=data["classification"],
        confidence=float(data["confidence"]),
        evidence=data.get("evidence", []),
    )


# <<Aggregation>>

def aggregate_results(results: list[PageResult]) -> tuple[str, float]:
    """
    Combine per-page verdicts into one document-level classification.

    Strategy: sum confidence scores for each class; the class with the
    higher total wins.
    """
    if not results:
        return "unknown", 0.0

    scanned = [r for r in results if r.classification == "scanned"]
    digital = [r for r in results if r.classification == "digital"]

    scanned_score = sum(r.confidence for r in scanned)
    digital_score = sum(r.confidence for r in digital)

    n = len(results)
    if scanned_score >= digital_score:
        winners = scanned
        overall = "scanned"
    else:
        winners = digital
        overall = "digital"

    if winners:
        confidence = (len(winners) / n) * (sum(r.confidence for r in winners) / len(winners))
    else:
        confidence = 0.0

    return overall, round(confidence, 4)


# <<Pipeline Orchestrator>>

def classify_pdf(pdf_path: str) -> DocumentResult:
    """
    Run the full VLM pipeline on every page of the PDF.

    Returns a DocumentResult with overall classification and per-page detail.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    client = anthropic.Anthropic()
    system = _build_system()   # cached across all page calls

    doc = fitz.open(str(path))
    total_pages = len(doc)
    page_results: list[PageResult] = []

    print(f"Classifying {total_pages} page(s) from: {path.name}")
    print(f"Model: {MODEL}\n")

    for i, page in enumerate(doc, start=1):
        print(f"  Page {i:>3}/{total_pages} ...", end=" ", flush=True)
        image_b64 = render_page_to_base64(page)
        result = classify_page(client, image_b64, i, system)
        page_results.append(result)
        print(f"{result.classification:<8}  confidence={result.confidence:.0%}")

    doc.close()

    overall_cls, overall_conf = aggregate_results(page_results)
    return DocumentResult(
        pdf_path=str(path),
        total_pages=total_pages,
        overall_classification=overall_cls,
        overall_confidence=overall_conf,
        per_page=page_results,
    )


# <<Print Report>>
def print_report(result: DocumentResult) -> None:
    """Print a structured human-readable report to stdout."""
    sep = "=" * 62
    thin = "─" * 62

    print(f"\n{sep}")
    print("  PDF CLASSIFICATION REPORT")
    print(sep)
    print(f"  File       : {result.pdf_path}")
    print(f"  Pages      : {result.total_pages}")
    print(f"  Result     : {result.overall_classification.upper()}")
    print(f"  Confidence : {result.overall_confidence:.1%}")
    print(f"\n{thin}")
    print("  PER-PAGE BREAKDOWN")
    print(thin)

    for pr in result.per_page:
        tag = f"  Page {pr.page_number:>3}"
        cls = pr.classification.ljust(8)
        conf = f"{pr.confidence:.0%}"
        print(f"{tag}  {cls}  conf={conf}")
        for ev in pr.evidence[:3]:          # show up to 3 evidence bullets
            print(f"            • {ev}")

    print(f"{sep}\n")


# <<Folder Processing>>

def classify_folder(folder_path: str) -> None:
    """Run the pipeline on every PDF found in folder_path and print each report."""
    folder = Path(folder_path)
    if not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        print(f"No PDF files found in: {folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(pdfs)} PDF(s) in: {folder}\n")

    summary: list[tuple[str, str, float]] = []  # (name, classification, confidence)

    for pdf in pdfs:
        try:
            result = classify_pdf(str(pdf))
            print_report(result)
            summary.append((pdf.name, result.overall_classification, result.overall_confidence))
        except json.JSONDecodeError as exc:
            print(f"  [SKIP] {pdf.name} — unparseable JSON from model: {exc}", file=sys.stderr)
        except anthropic.APIError as exc:
            print(f"  [SKIP] {pdf.name} — API error: {exc}", file=sys.stderr)

    # Print a final summary table across all docs
    sep = "=" * 62
    print(sep)
    print("  FOLDER SUMMARY")
    print(sep)
    for name, cls, conf in summary:
        print(f"  {name:<40}  {cls:<8}  {conf:.1%}")
    print(f"{sep}\n")


# <<CLI Entry Point>>

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify all PDFs in a folder as 'scanned' or 'digital' using a "
            "VLM-only pipeline powered by Claude Haiku."
        )
    )
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(DOCS_FOLDER),
        help=f"Folder containing PDFs to classify (default: {DOCS_FOLDER}).",
    )
    args = parser.parse_args()

    try:
        classify_folder(args.folder)
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
    