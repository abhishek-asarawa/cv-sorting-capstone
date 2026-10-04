import glob
import os
import sys

# Put the (bracketed) Codebase dir on sys.path so tests can import the modules.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in glob.glob(os.path.join(_HERE, "..", "Capstone_Project-CS*", "Codebase")):
    sys.path.insert(0, os.path.abspath(_p))
