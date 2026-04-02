import re

with open("turboquant/runtime/kv_interface.py", "r") as f:
    text2 = f.read()

target = """    def state(self) -> KVCacheState:
    def byte_size(self):
        return sum(b.packed_main.nbytes + b.scales.nbytes for b in self._blocks)

        return KVCacheState("""

replacement = """    def byte_size(self):
        return sum(b.packed_main.nbytes + b.scales.nbytes for b in self._blocks)

    def state(self) -> KVCacheState:
        return KVCacheState("""

text2 = text2.replace(target, replacement)

with open("turboquant/runtime/kv_interface.py", "w") as f:
    f.write(text2)
