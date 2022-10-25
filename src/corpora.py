"""
Tools for loading and processing corpora / materials sources (Visual Genome and Conceptual Captions).
"""

import itertools

import pandas as pd
from tqdm import tqdm

import logging
L = logging.getLogger(__name__)


def get_cooccurrences_for(cooccurrences, verbs, freq_thresh=2):
    """
    Extract relevant rows and columns for `verbs` from a co-occurrence matrix,
    consisting of verbs along the index and contexts along the columns. Returns
    a dataframe view where each column has nonzero total occurrences for some
    verb.
    """
    df = cooccurrences.reindex(list(verbs))
    drop_columns = df.columns[df.min(axis=0) < freq_thresh]
    return df.drop(columns=drop_columns)


def get_all_combinations(scene_occurrences, frame_occurrences,
                         min_scenes=2, max_scenes=2, min_frames=2, max_frames=2,
                         k_most_salient=10):
    """
    Get all possible combinations of verb1--verb2--scene set--frame set.

    Args:
        vg_relations:
        frames_df:
        min_scenes: Minimum size of a scene set.
        max_scenes: Maximum size of a scene set.
        min_frames: Minimum size of a frame set.
        max_frames: Maximum size of a frame set.
        k_most_salient: Evaluate just the top K scenes containing each verb
            pair, ranked by salience. (This filtering is necessary to make
            search efficient.)
    """

    # Get the intersection of the verbs covered in each corpus.
    all_verbs = set(scene_occurrences.index) & set(frame_occurrences.index)

    for verb1, verb2 in tqdm(list(itertools.combinations(all_verbs, 2)), desc="verb pairs"):
        # Retrieve relevant scene co-occurrence columms (excluding scenes where
        # neither verb is attested).
        scene_df = get_cooccurrences_for(scene_occurrences, [verb1, verb2])

        # Retrieve relevant frame co-occurrence columns (excluding frames where
        # neither verb is attested).
        frames_df = get_cooccurrences_for(frame_occurrences, [verb1, verb2])

        if len(scene_df.columns) < min_scenes or len(frames_df.columns) < min_frames:
            continue

        # Pre-calculate verb salience in each scene. We'll evaluate just the top K
        # scenes where the verbs are especially salient.
        salience_df = scene_occurrences[scene_df.columns]
        # Normalize so that we have one probability distribution per column (scene).
        salience_df = salience_df.div(salience_df.sum(axis=0), axis=1)
        # Compute minimum probability of verbs of interest in each scene.
        scene_saliences = salience_df.loc[[verb1, verb2]].min(axis=0)

        # Retain just the top K scenes ranked by v1--v2 salience.
        retain_scenes = list(scene_saliences.sort_values(ascending=False).head(k_most_salient).index)
        scene_df = scene_df[retain_scenes]

        tqdm.write("%s, %s" % (verb1, verb2))
        # tqdm.write(str(scene_df))
        # tqdm.write(str(frames_df))

        # Retrieve overlapping frames.
        scene_set_sizes = range(min_scenes, min(max_scenes + 1, len(scene_df.columns) + 1))
        frame_set_sizes = range(min_frames, min(max_frames + 1, len(frames_df.columns) + 1))
        for scene_set_size, frame_set_size in list(itertools.product(scene_set_sizes, frame_set_sizes)):
            scene_sets = itertools.combinations(scene_df.columns, scene_set_size)
            frame_sets = itertools.combinations(frames_df.columns, frame_set_size)

            yield verb1, verb2, scene_df, frames_df, salience_df, scene_sets, frame_sets
            
            
def load_and_preprocess_corpora(vg_relations_path, frames_path,
                                min_verb_freq=25, min_scene_freq=25, min_frame_freq=75,
                                ignore_verbs=("be", "have")):
    L.info("Loading VG relations.")
    vg_relations = pd.read_csv(vg_relations_path)

    # Strip sense information from vg labels.
    # TODO how bad is this?
    vg_relations["verb"] = vg_relations.verb_synset.str.split(".").str[0]

    L.info("Loading frame data.")
    frames_df = pd.read_csv(frames_path)

    # Drop unwanted verbs.
    vg_relations = vg_relations[~vg_relations.verb.isin(ignore_verbs)]
    frames_df = frames_df[~frames_df.lemma.isin(ignore_verbs)]

    # Verb frequency filtering
    if min_verb_freq is not None:
        vg_verbs_before = len(vg_relations.verb.unique())
        vg_verb_counts = vg_relations.verb.value_counts()
        drop_verbs = set(vg_verb_counts[vg_verb_counts < min_verb_freq].index)

        frames_verbs_before = len(frames_df.lemma.unique())
        frames_verb_counts = frames_df.lemma.value_counts()
        drop_verbs |= set(frames_verb_counts[frames_verb_counts < min_verb_freq].index)

        vg_relations = vg_relations[~vg_relations.verb.isin(drop_verbs)]
        frames_df = frames_df[~frames_df.lemma.isin(drop_verbs)]

        L.info("Dropped %i low-frequency verbs. %i remain in VG; %i remain in frames." %
                (len(drop_verbs), len(vg_relations.verb.unique()), len(frames_df.lemma.unique())))
        
    # Context frequency filtering
    if min_scene_freq is not None:
        vg_scenes_before = len(vg_relations.scene_id.unique())
        vg_scene_counts = vg_relations.scene_id.value_counts()
        drop_scenes = vg_scene_counts[vg_scene_counts < min_scene_freq].index
        vg_relations = vg_relations[~vg_relations.scene_id.isin(drop_scenes)]
        vg_scenes_after = len(vg_relations.scene_id.unique())

    if min_frame_freq is not None:
        frames_before = len(frames_df.children_str.unique())
        frame_counts = frames_df.children_str.value_counts()
        drop_frames = frame_counts[frame_counts < min_frame_freq].index
        frames_df = frames_df[~frames_df.children_str.isin(drop_frames)]
        frames_after = len(frames_df.children_str.unique())

    L.info("Dropped %i scenes (%i remaining) and %i frames (%i remaining) due to low frequency." %
            (vg_scenes_before - vg_scenes_after, vg_scenes_after,
             frames_before - frames_after, frames_after))

    L.info("Number of verbs shared between corpora after filtering: %i",
           len(set(vg_relations.verb) & set(frames_df.lemma)))
    
    return vg_relations, frames_df