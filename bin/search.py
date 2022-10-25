"""
Search for ideal verb pairs which are conflated in scene co-occurrence but can
be de-conflated with syntactic information.

Concretely, we are searching for pairs of verbs $(v_1, v_2)$ such that

1. there exist some set of scenes $S$ such that
    a. $p(v_1 \mid S) = \prod_i p(v_1 \mid S_i)$ has high variance.
       (The two are *conflated* in scene co-occurrence.)
    b. $p_{all}(v_1 \lor v_2 \mid S_i)$ is high on average.
       (The two are salient relative to other given verbs for the scene set.)
2. there exist some set of observed syntactic frames $F$ such that
    a. $p(v_1 \mid F) = \prod_i p(v_1 \mid F_i)$ has low variance.
       (The two are *de-conflated* with syntactic frame information.)
    b. TODO document total variation.

NB that the above probability distributions have support over just $v_1$ and
$v_2$, except for $p_{all}$, which is used to establish the salience of $v_1$
and $v_2$ among all attested verbs for each scene.
"""

from argparse import ArgumentParser
from contextlib import contextmanager
from functools import partial
import itertools
from multiprocessing import Pool, Queue
from multiprocessing.managers import SyncManager
import operator
from pathlib import Path
import queue
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

import logging
logging.basicConfig(level=logging.DEBUG)
L = logging.getLogger(__name__)


from corpora import get_cooccurrences_for, get_all_combinations, load_and_preprocess_corpora



# Prepare to share a push queue among processes.
class SearchManager(SyncManager):
    pass
class PriorityQueue(queue.PriorityQueue):
    def put_or_drop(self, item):
        with self.mutex:
            if self._qsize() == self.maxsize:
                # See if this candidate is better than the worst item.
                worst = self._get()
                if worst[0] < item[0]:
                    replacement = worst
                else:
                    replacement = item

                self._put(replacement)
            else:
                self._put(item)

    def get_all(self):
        with self.mutex:
            return list(self.queue)

    def get_maxsize(self):
        # Need to provide this as a method so that it can be accessed across
        # processes
        return self.maxsize

SearchManager.register("PriorityQueue", PriorityQueue)


def score_scene(scene_df, salience_df, verb1, verb2):
    """
    scene_df is a verb * scene df containing just two rows for verb1 and verb2
    salience_df is a verb * scene df containing all verb rows
    """
    # First component: variance of p(v1) per scene set. Maximize.
    var_v1_scene = scene_df.loc[verb1]
    var_v1_scene *= 1 - var_v1_scene
    var_v1_scene = var_v1_scene.mean()

    # Calculate salience of verb1, verb2 in scenes. Maximize.
    salience = salience_df.loc[[verb1, verb2]].min(axis=0).mean()

    return -var_v1_scene, -salience

def score_frame(frame_df, verb1, verb2):
    # First component: variance of p(v1) per frame set. Minimize.
    var_v1_frame = frame_df.loc[verb1]
    var_v1_frame *= 1 - var_v1_frame
    var_v1_frame = var_v1_frame.mean()

    # Second component: total variation of p(v1) across frame sets. Maximize.
    # This ensures that the frame set has distinguishing frames for each verb.
    total_variation = frame_df.loc[verb1].max() - frame_df.loc[verb1].min()

    return var_v1_frame, -total_variation


def worker_score(push_queue, candidates_desc):
    """
    Individual worker process. Pulls from a queue of candidates and pushes
    results to a limited priority queue.
    """
    verb1, verb2, scene_df, frame_df, salience_df, scene_sets, frame_sets = \
            candidates_desc

    scene_sets = list(scene_sets)
    frame_sets = list(frame_sets)

    # Normalize `scene_df` so that each column is a distribution over the two
    # verbs.
    scene_df = scene_df.div(scene_df.sum(axis=0), axis=1)

    # Normalize `frame_df` so that each column is a distribution over the two
    # verbs.
    frame_df = frame_df.div(frame_df.sum(axis=0), axis=1)

    #####
    # TODO magic number
    alpha = 0.5

    # Enumerate scene combinations and score.
    scene_scores = [score_scene(scene_df[list(scene_set)], salience_df[list(scene_set)],
                                verb1, verb2)
                    for scene_set in scene_sets]
    # TODO Magic number
    scene_scores = {scene_set: ((1 - alpha) * neg_scene_var + 0.5 * neg_scene_salience,
                                neg_scene_var, neg_scene_salience)
                    for scene_set, (neg_scene_var, neg_scene_salience)
                    in zip(scene_sets, scene_scores)}

    # Enumerate frame combinations and score.
    frame_vars = [score_frame(frame_df[list(frame_set)], verb1, verb2)
                  for frame_set in frame_sets]
    frame_scores = {frame_set: (alpha * frame_var + neg_total_variation,
                                frame_var, neg_total_variation)
                    for frame_set, (frame_var, neg_total_variation) in zip(frame_sets, frame_vars)}

    # Now combine scene + frame, only computing the top K.
    max_candidates = int(np.sqrt(push_queue.get_maxsize()))
    scene_scores = sorted(scene_scores.items(), key=operator.itemgetter(1))[:max_candidates]
    frame_scores = sorted(frame_scores.items(), key=operator.itemgetter(1))[:max_candidates]

    for (scene_set, scene_score), (frame_set, frame_score) in itertools.product(scene_scores, frame_scores):
        scene_score_comb, scene_score_var, scene_score_salience = scene_score
        frame_score_comb, frame_score_var, frame_score_negtv = frame_score
        total_score = scene_score_comb + frame_score_comb

        # Prepare an item for insertion into queue
        item_score = (total_score, scene_score_var, scene_score_salience,
                      frame_score_var, frame_score_negtv)
        item_descriptor = (verb1, verb2, tuple(scene_set), tuple(frame_set))
        item = (item_score, item_descriptor)
        push_queue.put_or_drop(item)



def main(args):
    vg_relations, frames_df = load_and_preprocess_corpora(
        args.vg_relations_path, args.frames_path,
        min_verb_freq=args.min_verb_freq,
        min_scene_freq=args.min_scene_freq,
        min_frame_freq=args.min_frame_freq,
        ignore_verbs=args.ignore_verbs)

    L.info("Calculating co-occurence data.")
    agg_func = max if args.cooccurrence_counter == "binary" else sum
    scene_cooccurrences = pd.get_dummies(vg_relations.scene_id).groupby(vg_relations.verb).apply(agg_func)
    frame_cooccurrences = pd.get_dummies(frames_df.children_str).groupby(frames_df.lemma).apply(agg_func)

    # Get all possible verb1--verb2--scene set--frame set combinations.
    all_combs = get_all_combinations(scene_cooccurrences, frame_cooccurrences)

    L.info("Beginning parallel search.")

    # Prepare for parallelized search.
    manager = SearchManager()
    manager.start()
    push_queue = manager.PriorityQueue(args.queue_size)

    def save_results():
        # Save results so far.
        df = pd.DataFrame([tuple(score) + tuple(descriptor) for score, descriptor in push_queue.get_all()],
                          columns=["score", "neg_var_v1_scene", "neg_salience",
                                   "var_v1_frame", "neg_frame_tv",
                                   "verb1", "verb2", "scene_set", "frame_set"])
        df = df.sort_values("score")
        df.to_csv(args.out_path)


    seen = set()
    with Pool(processes=args.n_jobs) as pool:
        run_worker = partial(worker_score, push_queue)
        for i, res in enumerate(pool.imap_unordered(run_worker, all_combs)):
            if i % args.save_every == 0:
                save_results()
                tqdm.write("Saved results at iteration %i." % i)

    save_results()

    manager.shutdown()


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("vg_relations_path", type=Path)
    p.add_argument("frames_path", type=Path)
    p.add_argument("-o", "--out_path", type=Path, default="search.csv")
    p.add_argument("-j", "--n_jobs", type=int, default=1)
    p.add_argument("-q", "--queue_size", type=int, default=500)
    p.add_argument("--ignore_verbs", type=lambda x: [v.strip() for v in x.strip().split(",")],
                   default=["be", "have"])
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--min_verb_freq", type=int, default=25)
    p.add_argument("--min_scene_freq", type=int, default=25)
    p.add_argument("--min_frame_freq", type=int, default=75)
    p.add_argument("--cooccurrence_counter", choices=["binary", "count"], default="count")

    main(p.parse_args())
