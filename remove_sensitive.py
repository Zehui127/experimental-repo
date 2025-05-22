import os

# Define keywords you want to remove
SENSITIVE_KEYWORDS = [""]  # Add more as needed

def sanitize_file(filepath, keywords):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    original_content = content

    for keyword in keywords:
        content = content.replace(keyword, "")

    if content != original_content:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"Sanitized: {filepath}")

def walk_and_sanitize(root_dir, keywords):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".py"):
                full_path = os.path.join(dirpath, filename)
                sanitize_file(full_path, keywords)

if __name__ == "__main__":
    ROOT_DIR = os.getcwd()  # Current directory
    walk_and_sanitize(ROOT_DIR, SENSITIVE_KEYWORDS)
