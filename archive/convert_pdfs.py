import os
from pathlib import Path
from pdfminer.high_level import extract_text
from dotenv import load_dotenv
load_dotenv()

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

def convert_pdf_to_txt(pdf_path: Path):
    txt = extract_text(str(pdf_path))
    out_path = pdf_path.with_suffix(".txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(txt)
    print(f"→ wrote {out_path}")

def main():
    pdfs = list(RAW_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in", RAW_DIR)
        return
    for pdf in pdfs:
        convert_pdf_to_txt(pdf)

if __name__ == "__main__":
    main()