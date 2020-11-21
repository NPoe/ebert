from wikipedia2vec import Dictionary
from wikipedia2vec.dump_db import DumpDB
from marisa_trie import Trie, RecordTrie
import copy
import joblib
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity_file", type = str, required = True)
    parser.add_argument("--src", type = str, required = True)
    parser.add_argument("--tgt", type = str, required = True)
    parser.add_argument("--dumpdb", type = str, required = True)

    args = parser.parse_args()

    dictionary = Dictionary.load(args.src)
    dumpdb = DumpDB(args.dumpdb)

    with open(args.entity_file) as handle:
        all_needed_entities_raw = set(handle.readlines())

    title2dest_title = dict(dumpdb.redirects())
    all_needed_entities = set([title2dest_title.get(title, title) for title in all_needed_entities_raw])

    src_file = joblib.load(args.src)

    old_word_dict = Trie()
    old_word_dict.frombytes(src_file['word_dict'])

    old_word_stats = src_file['word_stats']
    old_entity_stats = src_file['entity_stats']
    
    all_old_entities = [ent.title for ent in dictionary.entities()]
    all_old_entities_set = set(all_old_entities)

    all_new_entities = sorted([ent for ent in all_needed_entities if not ent in all_old_entities_set])
    joint_entity_stats = np.concatenate([old_entity_stats, 
        np.array([[5,5] for _ in all_new_entities]).astype(old_entity_stats.dtype)])

    new_entity_dict = Trie(all_old_entities + all_new_entities)

    new_redirect_dict = RecordTrie('<I', [
        (title, (new_entity_dict[dest_title],))
        for (title, dest_title) in dumpdb.redirects() if dest_title in new_entity_dict
    ])

    new_dictionary = Dictionary(\
            uuid=dictionary.uuid, 
            word_dict = old_word_dict, 
            entity_dict = new_entity_dict,
            redirect_dict = new_redirect_dict,
            word_stats = old_word_stats,
            entity_stats = joint_entity_stats,
            min_paragraph_len = dictionary.min_paragraph_len,
            language = dictionary.language,
            lowercase = dictionary.lowercase,
            build_params = dictionary.build_params)

    for entity in all_needed_entities_raw:
        assert new_dictionary.get_entity(entity) is not None

    for entity in all_needed_entities:
        assert new_dictionary.get_entity(entity) is not None
    
    for entity in all_old_entities:
        assert new_dictionary.get_entity(entity) is not None

    new_dictionary.save(args.tgt)
