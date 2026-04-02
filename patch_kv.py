import fifileinput
fileinput.input("turboquant/runtime/kv_interface.py", inplace=1)
for line in fileinput:
    if line.startswith("# Shim for mlx_lm"):
        line = "\n" + line + "\nTurboQuantKeysView:\n    """\n    Shim for mlx_lm compatibility.
    """\n"
    print("class " + line)
    continue
    print(line, end="")