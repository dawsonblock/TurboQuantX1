with open("turboquant/runtime/kv_interface.py", "r") as f:
    text = f.read()

text = text.replace("TurboQuantKeysView = EncodedKeyBlock", """
class TurboQuantKeysView:
    def __init__(self, cache, start: int, end: int):
        self.cache = cache
        self.start = start
        self.end = end
""")

with open("turboquant/runtime/kv_interface.py", "w") as f:
    f.write(text)
