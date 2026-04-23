import argparse
import time
import datetime
from pathlib import Path

from paddleocr import PaddleOCR

SCAN_DIR = Path("~/Documents/scans/inbox").expanduser()
DEST_DIR = Path("~/Documents/scans/processed").expanduser()
SUPPORTED_EXTS = {".pdf", ".jpg", ".jpeg", ".png"}

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_recognition_model_name="PP-OCRv5_mobile_rec",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False,
)


def extract_text(path: Path) -> str:
    result = ocr.predict(str(path))
    lines = []
    for page in result:
        lines.extend(page.json["res"]["rec_texts"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=Path, help="OCR a single file, print to stdout, and exit.")
    args = parser.parse_args()

    if args.file:
        print(extract_text(args.file))
        return

    DEST_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        for f in SCAN_DIR.iterdir():
            if f.suffix.lower() not in SUPPORTED_EXTS:
                continue

            text = extract_text(f)

            date = datetime.date.today().isoformat()
            org_path = DEST_DIR / f"{date}_{f.stem}.org"
            org_path.write_text(text)

            f.rename(DEST_DIR / f.name)
            print("Processed:", f.name)

        time.sleep(60)


if __name__ == "__main__":
    main()
