"""
This script is used to add the parent directory to the sys.path so that the modules 
in the parent directory can be imported in the scripts in the current directory.
"""

import os
import sys


def setup_paths():
    """
    Add the parent directory to the sys.path so that the modules in the parent directory can be imported
    """
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


setup_paths()
