import os
import re

def process_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    # Regular expression to match markdown links that start with http or https
    pattern = r'\[([^\]]+)\]\((https?://[^\)]+)\)(?!{:target="_blank"}{:rel="noopener noreferrer"})'
    
    # Replace links that don't have the target and rel attributes
    modified_content = re.sub(pattern, r'[\1](\2){:target="_blank"}{:rel="noopener noreferrer"}', content)

    if content != modified_content:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(modified_content)
        print(f"Modified: {file_path}")
    else:
        print(f"No changes needed: {file_path}")

def process_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.md'):
                file_path = os.path.join(root, file)
                process_file(file_path)

# Specify the directory containing your markdown files
posts_directory = '_posts'

# Process all markdown files in the specified directory
process_directory(posts_directory)