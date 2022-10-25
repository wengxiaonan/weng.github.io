"""
Extract syntactic frames from the Conceptual Captions corpus.
"""

from argparse import ArgumentParser
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

# Monkey-patch pandas to support progress monitoring
tqdm.pandas()

import logging
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)

nlp = None


def drop_none(xs):
    return [x for x in xs if x is not None]


def process_child(head_tok, child_tok):
    """
    Return a description of a head--child dependency relation relevant for
    syntactic frame analysis.
    """
    include_head_rels = ["prep", "prt"]
    # include prepositional head information
    if child_tok.dep_ in include_head_rels:
        return "%s:%s" % (child_tok.dep_, child_tok.text)

    return child_tok.dep_


def process_child_sequence(head_tok, children):
    """
    Post-process a sequence of parse children into a description of the head token's syntactic frame.
    """
    # Ignore child relations with punctuation, a priori irrelevant modifiers, etc.
    exclude_rels = ["punct", "mark", "amod", "advmod"]
    children = [child for child in children if child.dep_ not in exclude_rels]

    # Ignore conjoined clauses.
    if len(children) > 1 and children[-1].dep_ == "conj":
        children = children[:-1]
        # also remove the resulting cc if present
        if children[-1].dep_ == "cc":
            children = children[:-1]

    # Convert to deprel sequence.
    children = drop_none([process_child(head_tok, child) for child in children])

    # Collapse multiple contiguous instances of some relations into a single
    # child. (For example, we don't distinguish between a verb followed by one
    # adverb and a verb followed by two adverbs -- just that it is followed by
    # some nonzero number of adverbs.)
    ret, prev_child = [], None
    collapse_rels = ["amod", "advmod"]
    for child in children:
        if child in collapse_rels and child == prev_child:
            continue
        ret.append(child)
        prev_child = child

    return ret


def process_children(root_tok):
    left_children = process_child_sequence(root_tok, [tok for tok in root_tok.children if tok.i < root_tok.i])
    right_children = process_child_sequence(root_tok, [tok for tok in root_tok.children if tok.i > root_tok.i])
    return left_children, right_children


def get_root_children_str(row):
    """
    For the given caption row, get a string describing the syntactic frame of
    the root verb.
    """
    doc = nlp(row.caption)
    try:
        root_tok = next(tok for tok in doc if tok.pos_ == "VERB" and tok.dep_ == "ROOT")
    except StopIteration:
        return None, None

    left, right = process_children(root_tok)
    return root_tok.lemma_, "%s _ %s" % (" ".join(left), " ".join(right))


def process_df(df):
    df[["lemma", "children_str"]] = df.progress_apply(get_root_children_str, axis=1, result_type="expand")
    return df


def main(args):
    L.info("Loading spaCy model.")
    global nlp
    nlp = spacy.load(args.spacy)

    # Load data (may take a while).
    L.info("Loading input data.")
    df = pd.read_csv(args.input_path, sep="\t", header=None, names=["caption", "img"])

    # Prepare to parallelize.
    # Split dataframe into per-process chunks.
    df_split = np.array_split(df, args.num_cores)
    pool = Pool(args.num_cores)

    L.info("Processing syntactic frames.")
    df = pd.concat(pool.map(process_df, df_split))

    pool.close()
    pool.join()

    # Drop rows without lemmata.
    df = df[~df.lemma.isna()]

    df.to_csv(args.output_path)


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("input_path", type=Path)
    p.add_argument("-o", "--output_path", type=Path, default=Path("frames.csv"))
    p.add_argument("--spacy", default="/home/jgauthie/.local/lib/python3.6/site-packages/en_core_web_sm/en_core_web_sm-2.0.0")
    p.add_argument("-j", "--num_cores", type=int, default=1)

    args = p.parse_args()
    main(args)
