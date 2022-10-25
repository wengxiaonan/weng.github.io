"""
Extract verb--scene co-occurrence information from Visual Genome.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Monkey-patch pandas to support progress monitoring
tqdm.pandas()

import logging
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)


def load_verb_relations(vg_path):
    with (vg_path / "relationships.json").open("r") as rel_f:
        vg_rel = json.load(rel_f)

    verb_relations = []
    for scene in tqdm(vg_rel):
        for relationship in scene["relationships"]:
            try:
                verb = next(syn for syn in relationship["synsets"] if ".v." in syn)
            except StopIteration: continue

            verb_relations.append((
                scene["image_id"], verb,
                relationship["subject"]["object_id"],
                relationship["object"]["object_id"]))

    verb_relations = pd.DataFrame(verb_relations,
            columns=["scene_id", "verb_synset", "subject_id", "object_id"])
    return verb_relations


def main(args):
    verb_relations = load_verb_relations(args.vg_path)
    verb_relations.to_csv(args.out_path)



if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("vg_path", type=Path)
    p.add_argument("-o", "--out_path", type=Path, default=Path("vg_relations.csv"))

    args = p.parse_args()
    main(args)
