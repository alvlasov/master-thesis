import sys
import html
import logging
import shelve
import concurrent.futures

import json
import re

from operator import itemgetter
from collections import Counter
from datetime import datetime, timedelta
from importlib import reload

import pandas as pd
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import datefinder
from tqdm import tqdm_notebook, tqdm
import joblib
import anytree

pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', 100)
np.set_printoptions(precision=4, suppress=True, sign=' ')

import east

def execute(func, iterator, n_jobs=1):
    futures = []
    
    executor = concurrent.futures.ProcessPoolExecutor(n_jobs)
    
    for i in iterator:

        if len(futures) == n_jobs:
            d_futures = concurrent.futures.wait(futures, return_when='FIRST_COMPLETED')

            for f in d_futures.done:
                yield f.result()

            futures = list(d_futures.not_done)

        f = executor.submit(func, i)
        futures.append(f)
        
    for f in concurrent.futures.as_completed(futures):
        yield f.result()