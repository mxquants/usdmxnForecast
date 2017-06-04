# -*- coding: utf-8 -*-
"""
Get imports according to OS
@author: Rodrigo Hernández-Mota
"""

import os
import platform


def getUserDir():
    """Change user to personalized dir according to OS."""
    operative_system = platform.system().lower()
    if 'win' in operative_system:
        print("\nHi there, you are probably Danny. Else, change imports.py\n")
        os.chdir("")
    if "lin" in operative_system:
        print("\nHi there, you are probably Ro. Else, change imports.py\n")
        os.chdir("/media/rhdzmota/Data/Files/github_mxquants/usdmxnForecast")


def getImports():
    """Import imports."""
    getUserDir()
    imports = """\
import numpy as np
import pandas as pd
import quanta as mx
import datetime as dt
import matplotlib.pyplot as plt
from metallic_blue_lizard.neural_net import competitive_neurons
"""
    return imports
