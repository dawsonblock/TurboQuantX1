import os

files = [
    'tests/unit/test_attention_score_block.py',
    'tests/unit/test_attention_score_block_qjl.py',
    'tests/unit/test_decode_k_block.py',
    'tests/unit/test_encoded_key_block.py',
]

for file in files:
    if os.path.exists(file):
        with open(file, 'r') as f:
            content = f.read()
        
        content = content.replace("KVCompressor", "TurboQuantKVCache")
        content = content.replace("TurboQuantKeysView", "Any")
        
        with open(file, 'w') as f:
            f.write(content)
        print(f"Patched {file}")
