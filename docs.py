"""
Script for documentation generation and cleaning.
"""

import glob
import subprocess
import os

files_before_generation = glob.glob("./docs/source/*.rst")
subprocess.call(["sphinx-apidoc", "-f", "-o", "docs/source", "./cortex"])
subprocess.call(["make", "html"])
files_after_generation = glob.glob("./docs/source/*.rst")

for filename in files_before_generation:
    if filename in files_after_generation:
        files_after_generation.remove(filename)

for filename in files_after_generation:
    subprocess.call(["rm", "-f", filename])

folders = [
    folder for folder in os.listdir('./docs/html')
    if os.path.isdir(os.path.join('./docs/html', folder))
]

folders_to_remove = []

for folder in folders:
    if folder[0] != "_":
        folders_to_remove.append(folder)

for folder_to_remove in folders_to_remove:
    folder_to_remove = "./docs/html/" + folder_to_remove
    subprocess.call(["rm", "-rf", folder_to_remove])

doctrees = glob.glob("./docs/doctrees")
subprocess.call(["rm", "-rf", doctrees[0]])

sources = glob.glob("./docs/html/_sources")
subprocess.call(["rm", "-rf", sources[0]])

objects = glob.glob("./docs/html/objects.inv")
subprocess.call(["rm", "-f", objects[0]])
