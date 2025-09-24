import os


def read_file_with_fallback(path):
    """Reads the file with fallback encoding, helper function for the loading of image content"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except UnicodeDecodeError:
        with open(path, "r", encoding="iso-8859-1") as f:
            return f.read().strip()
def read_file_to_list(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = [line.strip() for line in file if line.strip()]  # remove empty lines
    return lines

def load_image_content(bug_dir, bug_id):
    """Loads all image content files for a given bug report, helper function for the extended query"""
    content = ""
    image_files = sorted([
        f for f in os.listdir(bug_dir)
        if f.startswith(bug_id) and f.endswith("ImageContent.txt")
    ])
    for image_file in image_files:
        image_path = os.path.join(bug_dir, image_file)
        content += "\n" + read_file_with_fallback(image_path)
    return content.strip()

def load_image_URLs(bug_dir, bug_id):
    """Extract the image URLs' from imageURL.txt file"""
    URL_content = []
    image_path=os.path.join(bug_dir,"imageURL.txt")
    URL_content = read_file_to_list(image_path)
    return URL_content;


