import sys
import os

# Allow `import src.data.generator` from repo root without manual path hacks
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
