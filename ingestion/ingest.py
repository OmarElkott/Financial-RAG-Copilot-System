from pathlib import Path
from bs4 import BeautifulSoup
import json


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "tech"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

def extract_metadata(file_path: Path, text_blocks, full_text) -> dict:
    # name = file_path.name

    # Defaults
    ticker = "UNKNOWN"
    company = "UNKNOWN"
    document_type = "UNKNOWN"
    fiscal_period = "UNKNOWN"
    
    if "Apple" in file_path.name:
        ticker = "AAPL"
        company = "Apple"
    elif "Microsoft" in file_path.name:
        ticker = "MSFT"
        company = "Microsoft"
    
    if "10-K" in file_path.name:
        document_type = "10-K"
    elif "ETC" in file_path.name:
        document_type = "ETC"                               
        
    if "Q4_2025" in file_path.name:
        fiscal_period = "Q4_2025"
    elif "2025" in file_path.name:
        fiscal_period = "2025"
        
    document_id = f"{ticker.lower()}_{document_type.lower()}_{fiscal_period}_{file_path.stem}"
        
    # TODO: parse filename or HTML for company, ticker, document_type, etc.
    return {
        "document_id": document_id,
        "ticker": ticker,
        "source_uri": file_path.name,
        "document_type": document_type,
        "fiscal_period": fiscal_period,
        "company": company,
        "file_name": file_path.name,
        "content": text_blocks,
        "full_text": full_text,
    }

def parse_html(file_path: Path) -> dict:
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f, "html.parser")

    # Remove unwanted tags
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    text_blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "table"]):
        text = tag.get_text(separator=" ", strip=True)
        if text:
            text_blocks.append({"type": tag.name, "text": text})

    full_text = " ".join(block["text"] for block in text_blocks)
    metadata = extract_metadata(file_path, text_blocks, full_text)
    
    return metadata

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "documents.jsonl"

    with output_path.open("w", encoding="utf-8") as out:
        for file_path in DATA_DIR.glob("*.html*"):
            print(f"Processing: {file_path.name}")
            try:
                record = parse_html(file_path)
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"Failed on {file_path.name}: {e}")

if __name__ == "__main__":
    main()