import sys
import pathlib

# tests/ je podadresár — pridaj root projektu do sys.path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
