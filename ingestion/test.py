from pathlib import Path
from bs4 import BeautifulSoup

html_files = []

# 1. Get the directory of main.py
script_dir = Path(__file__).resolve().parent

# 2. Go up to the root folder, then down into data/raw/tech
html_folder = script_dir.parent / "data" / "raw" / "tech"

def read_files(): 
    # 3. Loop through and read every HTML file in that folder
    for file_path in html_folder.glob("*.html*"):
        print(f"Opening: {file_path.name}")
        
        # FIX 1: Added errors="ignore" to bypass corrupted/non-UTF-8 bytes safely
        with file_path.open("r", encoding="utf-8", errors="ignore") as file:
            soup = BeautifulSoup(file, "html.parser")
            html_files.append(soup)

def parse_files():
    # FIX 2: Loop through the list of soup objects first, then call find_all on each soup
    for soup in html_files:
        for element in soup.find_all(["h1", "p", "a"]):
            # Use strip() to clean up messy HTML whitespace/newlines
            text = element.text.strip()
            if text:  # Only print if there is actual content
                print(element.name, "->", text)

def main(): 
    read_files()
    parse_files()
    
if __name__ == "__main__":
    main()
