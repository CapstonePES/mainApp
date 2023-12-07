import pandas as pd
import numpy as np
import random


def lstm(a):
    a= a[1:]
    # DUMMY FUNCTION REMOVE LATER!!!!!!!!
    a = [(i/8)*random.uniform(0.8,1.0) for i in a]
    s = sum(a)*10 + 20
    if s > 95:
        s = 95.231
    return s
