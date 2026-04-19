import re

with open('train_dl_enhanced_model.py', 'r', encoding='utf-8', errors='ignore') as f:
    content = f.read()

# Find the last valid print statement
match = re.search(r'print\("    • overall_flowchart\.png"\)', content)
if match:
    end_pos = match.end()
    clean_content = content[:end_pos] + '\n'
    with open('train_dl_enhanced_model_clean.py', 'w', encoding='utf-8') as f:
        f.write(clean_content)
    print('File cleaned successfully')
else:
    print('Pattern not found')