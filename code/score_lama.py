import json
import argparse
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type = str, required = True)
    args = parser.parse_args()

    for method in ("bert", "linear_replace", "linear_concat"):
        print(method)
        ranks_by_relation = {}
        with open(args.file) as handle:
            for line in handle:
                obj = json.loads(line)
                relation = obj.get("pred", obj["predicate_id"])
                if not relation in ranks_by_relation:
                    ranks_by_relation[relation] = []

                ranks_by_relation[relation].append(obj["gold_rank:" + method])

        for relation in ranks_by_relation:
            ranks_by_relation[relation] = np.array(ranks_by_relation[relation])

        for k in (1, 2, 5, 10):
            print(f"Hits@{k} = ", np.mean([(ranks_by_relation[relation] <= k).mean() for relation in ranks_by_relation]))
        
        print("*****")
