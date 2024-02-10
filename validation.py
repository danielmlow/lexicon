
import sys
import pandas as pd
import numpy as np
sys.path.append('./../../concept-tracker/')
from concept_tracker.lexicon import load_lexicon


srl = load_lexicon('./data/input/lexicons/suicide_risk_lexicon_calibrated_unmatched_tokens_unvalidated_24-02-08T23-14-18.pickle')

# TODO: move last part of calibration here? Or just load validated lexicons here?