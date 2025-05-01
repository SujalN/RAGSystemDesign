import pdfplumber
from pathlib import Path

RAW_DIR   = Path(__file__).parent.parent / "data" / "raw"
TABLE_DIR = Path(__file__).parent.parent / "data" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

def cell_to_str(cell):
    """Convert a table cell value to a clean string."""
    if cell is None:
        return ""
    return str(cell).strip()

def table_to_markdown(table):
    # Sanitize header and rows
    header, *rows = table
    header = [cell_to_str(c) for c in header]
    md  = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join("---" for _ in header) + " |\n"
    for row in rows:
        row = [cell_to_str(c) for c in row]
        md += "| " + " | ".join(row) + " |\n"
    return md

def main():
    for pdf_path in RAW_DIR.glob("*.pdf"):
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                tables = page.extract_tables()
                if tables:
                    print(f"{pdf_path.name} page {i}: found {len(tables)} table(s)")
                for j, table in enumerate(tables, start=1):
                    # Skip trivial tables
                    if not table or len(table) < 2:
                        continue
                    md = table_to_markdown(table)
                    out = TABLE_DIR / f"{pdf_path.stem}_p{i}_tbl{j}.md"
                    out.write_text(md, encoding="utf-8")
                    print(f"  → wrote table: {out.name}")

if __name__ == "__main__":
    main()
