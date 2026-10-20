from utils.style_extractor import extract_style

transcript = open("data/ami_transcripts/sample.txt").read()
print(extract_style(transcript))