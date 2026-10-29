"""Manual live smoke check: python test_style_extractor.py (uses API credits)."""
from pathlib import Path
from utils.style_extractor import extract_style

if __name__ == '__main__':
    transcript = (Path(__file__).parent / 'data/ami_transcripts/sample.txt').read_text()
    print(extract_style(transcript))
