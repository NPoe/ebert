import argparse
import sys
import numpy as np

def load_file(f):
    spans = {}
    with open(f) as handle:
        for line in handle:
            splitline = line.strip().split("\t")
            start, end = int(splitline[0]), int(splitline[1])
            assert not (start, end) in spans
            spans[(start, end)] = splitline[2:]

    return spans


def score_macro_f1(true, pred, show = False):
    true_by_ent = {}
    pred_by_ent = {}

    for span, x in true.items():
        if not x[0] in true_by_ent:
            true_by_ent[x[0]] = {}
        true_by_ent[x[0]][span] = x
    
    for span, x in pred.items():
        if not x[0] in pred_by_ent:
            pred_by_ent[x[0]] = {}
        pred_by_ent[x[0]][span] = x
    
    precs, recs = [], []
    for ent in set(true_by_ent.keys()).union(set(pred_by_ent.keys())):
        prec, rec, f1 = score_micro_f1(true_by_ent.get(ent, {}), pred_by_ent.get(ent, {}))[:3]
        precs.append(prec)
        recs.append(rec)

    macro_prec = np.mean(precs)
    macro_rec = np.mean(recs)
    macro_f1 = macro_prec * macro_rec * 2 / (macro_prec + macro_rec) if macro_prec + macro_rec else 0.0

    return macro_prec, macro_rec, macro_f1

def score_micro_f1(true, pred, show = False):
    strong_matches = list(set(true.keys()).intersection(set(pred.keys())))
    strong_matches_correct = [span for span in strong_matches if true[span][0] == pred[span][0]]
    
    prec = 1.0 if len(pred) == 0 else len(strong_matches_correct) / len(pred)
    rec = 1.0 if len(true) == 0 else len(strong_matches_correct) / len(true)
    f1 = 0.0 if (prec+rec) == 0 else 2*prec*rec/(prec+rec)

    if show:
        true_show = [(key[0], key[1], value, "true") for key, value in true.items() if not key in strong_matches_correct]
        true_pred = [(key[0], key[1], value, "pred") for key, value in pred.items() if not key in strong_matches_correct]
        together = sorted(true_show + true_pred)
        for i in together:
            print(i)

    return prec, rec, f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type = str, required = True)
    parser.add_argument("--pred", type = str, required = True)
    args = parser.parse_args()

    true = load_file(args.gold)
    pred = load_file(args.pred)

    print("Micro P: {:.5} R: {:.5} F1: {:.5}".format(*score_micro_f1(true, pred)))
    print("Macro P: {:.5} R: {:.5} F1: {:.5}".format(*score_macro_f1(true, pred)))
