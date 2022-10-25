"""
Utilities for loading and preprocessing Psiturk experimental data.
"""

from copy import copy
import json
import logging
import sqlite3

import pandas as pd

L = logging.getLogger(__name__)


PSITURK_DB_PATH = "psiturk/participants.db"
PSITURK_DATA_TABLE = "turkdemo"


# Allows us to return sqlite rows as dicts rather than tuples.
def dict_factory(cursor, row):
    d = {}
    for idx, col in enumerate(cursor.description):
        d[col[0]] = row[idx]
    return d

def _get_connection(db_path):
    if not hasattr(_get_connection, "conn"):
        _get_connection.conn = sqlite3.connect(db_path)
        _get_connection.conn.row_factory = dict_factory
        
    return _get_connection.conn

def c():
    if not hasattr(c, "cur"):
        c.cur = _get_connection(PSITURK_DB_PATH).cursor()
    return c.cur

def load_raw_results():
    return pd.DataFrame(c().execute(f"SELECT * FROM {PSITURK_DATA_TABLE}")).set_index("uniqueid")

def get_trials_df(raw_results, extract_data_fields=()):
    """
    Split raw data into a data frame which has one row per subject--trial.
    """
    trials = []
    for uid, row in raw_results.iterrows():
        # TODO process other status codes
        if pd.isna(row.datastring):
            L.warn("Missing datastring for uid %s. Status was %i." % (uid, row.status))
            continue
            
        data = json.loads(row.datastring)
        
        base_info = {c: data[c] for c in ["condition", "counterbalance", "assignmentId", "workerId", "hitId"]}
        base_info["uniqueid"] = uid
        
        for trial in data["data"]:
            tdata = trial["trialdata"]
            info = copy(base_info)
            # Extract generally useful trial-specific data
            info.update({c: tdata[c] for c in ["trial_type", "trial_index", "rt", "internal_node_id"]})
            # Extract user-specified trial data
            info.update({c: tdata.get(c, None) for c in extract_data_fields})
            
            # Process responses from survey plugin.
            if tdata["trial_type"].startswith("survey"):
                # Add a single row per survey question.
                responses = json.loads(tdata["responses"])
                for idx in range(len(responses)):
                    key = "Q%i" % idx
                    
                    idx_info = copy(info)
                    idx_info["survey_question_idx"] = idx
                    idx_info["survey_answer"] = responses[key]
                    trials.append(idx_info)
            else:
                trials.append(info)
            
    return pd.DataFrame(trials).astype({"survey_question_idx": pd.Int64Dtype()}) \
        .set_index(["trial_index", "uniqueid"])