def read_text_file(file_path):
    """Reads the contents of a text file and returns it as a string."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_text_file(file_path, content):
    """Writes the given content to a text file."""
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)