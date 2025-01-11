'''
Tokenizes the scraped pdf file from the spectrum portal. Cannot be
run as a stand-alone program - gets executed through the spider program 
when a valid pdf is found. 
'''
import os
import re
from collections import defaultdict
from PyPDF2 import PdfReader
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

cwd = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.realpath(cwd))

# Load stop words
stop_words = set(stopwords.words("english"))
INDEX = "/".join([root, 'INDEX'])
TOKENS = "/".join([INDEX,'tokens.txt'])

def compress(tokens):
    """
    Compresses tokens by removing stopwords, punctuation, applying lowercase,
    and removing non-printable ASCII characters.
    """
    ascii_chars = re.compile(r"[\x20-\x7E]+")
    compressed_tokens = [
        token.lower() for token in tokens
        if token.lower() not in stop_words
        and re.match(ascii_chars, token)  # Filter printable ASCII
        and re.match(r"^\w+$", token)     # Only alphanumeric tokens
    ]
    return compressed_tokens

def tokenize_pdf_file(file_path):
    """
    Tokenizes a PDF file and saves the tokens to a file.

    Args:
        file_path (str): Path to the PDF file.
    """
    if not os.path.exists(INDEX):
        os.mkdir(INDEX)
    try:
        # Read PDF content
        reader = PdfReader(file_path)
        text_content = []

        for page in reader.pages:
            text_content.append(page.extract_text() or "")

        # Combine all pages into one string
        text = " ".join(text_content)

        # Tokenize and compress tokens
        tokens = word_tokenize(text)
        compressed_tokens = compress(tokens)

        # Track positions of each token
        term_occurrences = defaultdict(list)
        for index, token in enumerate(compressed_tokens):
            term_occurrences[token].append(index)

        # Collect positional postings for this document
        file_name = os.path.basename(file_path)  # Extracts the file name from the file path
        document_id = int(file_name.split("_")[0])  # Document ID from filename
        document_postings = [
            (term, document_id, positions) for term, positions in term_occurrences.items()
        ]

        print(f"Tokenized {file_path}: {len(document_postings)} tokens with positions")

    except Exception as e:
        print(f"Error processing {file_path}: {e}")

    with open(TOKENS, "a", encoding="utf-8") as f:  # Append mode

        f.write(f"{document_postings}\n")
