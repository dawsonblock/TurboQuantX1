import os
import subprocess
import shutil

master = "/Users/dawsonblock/Downloads/Turbo-master"
rc = "/Users/dawsonblock/Downloads/TurboQuant-release-candidate-main 2"
opts = [
    "rsync", "-a", 
    "--exclude", ".git", 
    "--exclude", "__pycache__", 
    "--exclude", ".ruff_cache", 
    "--exclude", ".pytest_cache", 
    "--exclude", ".DS_Store", 
    "--exclude", "README.md"
]

dirs = ["turboquant/", "integrations/", "tests/", "scripts/", "tools/", "docs/"]
for d in dirs:
    subprocess.run(opts + [os.path.join(master, d), os.path.join(rc, d)], check=True)

subprocess.run(opts + [os.path.join(master, "mlx_lm/models/cache.py"), os.path.join(rc, "mlx_lm/models/cache.py")], check=True)
shutil.copy(os.path.join(master, "noxfile.py"), os.path.join(rc, "noxfile.py"))

# Show what changed
subprocess.run(["git", "status", "--short"], cwd=rc)
