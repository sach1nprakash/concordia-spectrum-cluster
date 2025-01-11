'''
Builds an inverted positional index for the tokenized pdf files.
Can be run as a standalone program - provided the tokens are available 
in the expected format. Use command python positional_index.py
'''
import os
import re

cwd = os.path.dirname(os.path.realpath(__file__))
root = os.path.dirname(os.path.realpath(cwd))

# Global Constants
OUTPUT_DIRECTORY = "/".join([root, 'INDEX'])
BLOCK_PREFIX = "BLOCK"
BLOCK_SUFFIX = ".txt"
OUTPUT_POSITIONAL_INDEX = "index"
OUTPUT_POSITIONAL_INDEX = "/".join([OUTPUT_DIRECTORY,
                                   OUTPUT_POSITIONAL_INDEX + BLOCK_SUFFIX])
LIST_OF_TOKEN_BLOCKS = []
OUTPUT_TOKENS = "/".join([OUTPUT_DIRECTORY, 'tokens' + BLOCK_SUFFIX])


def load_tokens():
    """
    Reads and loads the existing tokens from tokens.txt into LIST_OF_TOKEN_BLOCKS.
    :return: list of set of tokens loaded from disk.
    """
    print("Loading pre-processed tokens from tokens.txt...")
    global number_of_tokens, number_of_documents, average_document_length, LIST_OF_TOKEN_BLOCKS

    try:
        with open(OUTPUT_TOKENS, 'r', errors='ignore', encoding='utf-8') as fp:
            # Read each line, evaluate it to get a list, and add it to LIST_OF_TOKEN_BLOCKS
            LIST_OF_TOKEN_BLOCKS = [eval(line.strip()) for line in fp]

            # Calculate the number of tokens and documents
            number_of_tokens = sum(len(token_list)
                                   for token_list in LIST_OF_TOKEN_BLOCKS)
            number_of_documents = len(LIST_OF_TOKEN_BLOCKS)
            average_document_length = number_of_tokens / \
                number_of_documents if number_of_documents else 0

            print(
                f"Loaded {number_of_documents:,} documents and {number_of_tokens:,} tokens from tokens.txt.\n")
    except FileNotFoundError:
        print(f"Error: {OUTPUT_TOKENS} not found.")
        LIST_OF_TOKEN_BLOCKS = []
        number_of_tokens = 0
        number_of_documents = 0
        average_document_length = 0

    return LIST_OF_TOKEN_BLOCKS


def init_pos_indexing():
    """
    Initialize the positional indexing by setting up the output directory and loading tokens.
    """
    global LIST_OF_TOKEN_BLOCKS
    prepare_output_directory(OUTPUT_DIRECTORY)
    LIST_OF_TOKEN_BLOCKS = load_tokens()


def prepare_output_directory(output_directory):
    """
    Create an output directory where disk blocks will be stored. If it already exists, 
    clear its contents.
    :param output_directory: Directory where block files will be written.
    """
    try:
        if not os.path.exists(output_directory):
            os.mkdir(output_directory)
        else:
            for file in os.listdir(output_directory):
                if file != 'tokens.txt':
                    os.unlink(os.path.join(output_directory, file))
    except FileExistsError:
        pass


def add_to_dictionary(dictionary, term):
    """
    Add a new term to the dictionary with an empty postings list.
    :param dictionary: Dictionary mapping terms to postings.
    :param term: The term to add to the dictionary.
    :return: The newly created postings list for the term.
    """
    dictionary[term] = []
    return dictionary[term]


def add_to_postings_list(postings_list, document_id, document_positions):
    """
    Add a document ID and positions to the term's postings list.
    :param postings_list: List of postings for the term.
    :param document_id: Document ID where the term appears.
    :param document_positions: List of positions where the term appears in the document.
    """
    postings_list.append({document_id: document_positions})


def sort_terms(dictionary):
    """
    Get the terms from the dictionary and sort them alphabetically.
    :param dictionary: Dictionary mapping terms to postings.
    :return: A list of alphabetically sorted terms.
    """
    return sorted(dictionary)


def write_block_to_output_directory(sorted_terms, dictionary, block_file):
    """
    Write sorted terms and their postings to a block file in the output directory.
    :param sorted_terms: Alphabetically sorted list of terms.
    :param dictionary: Dictionary mapping terms to postings.
    :param block_file: File path where the data will be written.
    """
    with open(block_file, encoding='utf-8', mode='w') as file:
        for term in sorted_terms:
            postings_list = dictionary[term]
            formatted_postings = ' '.join(
                [f"{doc_id}[{','.join(map(str, positions))}]" for posting in postings_list for doc_id,
                 positions in posting.items()]
            )
            line = f"{term} {formatted_postings}\n"
            file.write(line)


def construct_pos_index():
    """
    Construct a positional index from tokenized documents.
    :return: The merged index dictionary.
    """

    if os.path.exists(OUTPUT_POSITIONAL_INDEX):
        return get_index()

    block_files = []
    block_number = 0

    for list_of_tokens in LIST_OF_TOKEN_BLOCKS:
        data = {}
        for token in list_of_tokens:
            term, doc_id, positions = token[0], token[1], token[2]

            postings_list = data.get(term, add_to_dictionary(data, term))
            add_to_postings_list(postings_list, doc_id, positions)

        block_number += 1
        sorted_terms = sort_terms(data)
        block_file = f"{OUTPUT_DIRECTORY}/{BLOCK_PREFIX}{block_number}{BLOCK_SUFFIX}"
        write_block_to_output_directory(sorted_terms, data, block_file)
        block_files.append(block_file)

    return merge_blocks(block_files)


def merge_blocks(block_files):
    """
    Merge multiple block files into a final index.
    :param block_files: List of block file paths.
    :return: The final merged index as a dictionary.
    """
    block_streams = [open(block_file, errors='ignore', encoding='utf-8')
                     for block_file in block_files]
    lines = [block.readline().strip() for block in block_streams]
    most_recent_term = ""

    with open(OUTPUT_POSITIONAL_INDEX, encoding='utf-8', mode='w') as output_index:
        while block_streams:
            # Filter out empty lines
            valid_lines = [(i, line)
                           for i, line in enumerate(lines) if line.strip()]
            if not valid_lines:
                break

            # Find the minimum line based on the term
            min_index, _ = min(valid_lines, key=lambda x: x[1].split()[0])
            line = lines[min_index]
            current_term = line.split()[0]

            # Extract postings from the line
            matches = re.findall(r'(\d+)\[([\d, ]+)\]', line)
            current_postings = " ".join(
                [f"{doc_id}[{positions.strip()}]" for doc_id, positions in matches])

            if current_term != most_recent_term:
                output_index.write(f"\n{current_term} {current_postings}")
                most_recent_term = current_term
            else:
                output_index.write(f" {current_postings}")

            # Read the next line from the current block
            lines[min_index] = block_streams[min_index].readline().strip()
            if not lines[min_index]:
                block_streams[min_index].close()
                block_streams.pop(min_index)
                lines.pop(min_index)

    return get_index()


def get_index():
    """
    Load the index from the merged index file and return it as a dictionary.
    :return: The inverted index as a dictionary.
    """
    inverted_index = {}

    with open(OUTPUT_POSITIONAL_INDEX, encoding='utf-8') as index_file:
        index_file.readline()  # Skip the first empty line
        for line in index_file:
            term, postings_str = line.split(maxsplit=1)
            matches = re.findall(r'(\d+)\[([\d, ]+)\]', postings_str)
            postings = [{int(doc_id): list(map(int, positions.split(',')))}
                        for doc_id, positions in matches]
            inverted_index[term] = postings

    return inverted_index


def word_locations(file_name, word):
    """
    Find the locations of a word in a file.
    :param file_name: The file to search in.
    :param word: The word to find locations of.
    :return: List of positions where the word occurs.
    """
    with open(file_name, encoding='utf-8', mode='r') as file:
        data = file.read()
    return [i for i, x in enumerate(data.split()) if x == word]


if __name__ == "__main__":
    init_pos_indexing()
    construct_pos_index()
