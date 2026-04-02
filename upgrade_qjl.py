import sys

def modify_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()
    
    for old, new in replacements:
        if old in content:
            content = content.replace(old, new)
        else:
            print(f"Warning: Could not find '{old[:30]}...' in {path}")
            
    with open(path, "w") as f:
        f.write(content)

# We will handle these files manually via terminal or scripting.
