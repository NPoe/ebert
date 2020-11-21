import numpy as np
import argparse
from embeddings import *
from mappers import *

def train_mapper(src, tgt, pairs, mapper, train_args):
    x = src[[i for i,_ in pairs]]
    y = tgt[[j for _,j in pairs]]
    mapper.train(x, y, **train_args)

    return None

def train_mapper_em(src, tgt, pairs, mapper, em_mapper,
        train_args, em_train_args, sample_size, iterations, 
        normalize=False, hard_em=True, increase_sample_size=True,
        seed=0, verbose=1, epsilon = 1e-7):
   
    if em_mapper is None:
        em_mapper = mapper
    
    if em_train_args is None:
        em_train_args = train_args

    x = src[[i for i,_ in pairs]]
    y = tgt[[j for _,j in pairs]]

    em_mapper.train(x, y, w=None, **em_train_args)

    N, D = y.shape    
    sample_size = min(D, sample_size) if sample_size else D

    alpha = 0.5
    
    mu_y = np.mean(y, axis = 0)
    var_y = np.var(y, axis = 0)
    var_x = np.var(em_mapper.apply(x) - y, axis = 0)
    
    all_dimensions = list(range(D))
    state = np.random.RandomState(seed)

    if increase_sample_size:
        sample_size_step = (D - sample_size)/iterations
        sample_sizes = [sample_size + int(i*sample_size_step) for i in range(1,iterations+1)]
    else:
        sample_sizes = [sample_size] * iterations

    prev_alpha = 0
    for iteration in range(1, iterations+1):
        if abs(prev_alpha - alpha) < epsilon and sample_size == D:
            break
        prev_alpha = alpha

        # E Step
        sample_size = sample_sizes[iteration-1]
        dims = state.choice(all_dimensions, (sample_size,), replace = False)
        mu_x = em_mapper.apply(x)

        sqdiff_x = (y[:,dims] - mu_x[:,dims])**2
        sqdiff_y = (y[:,dims] - mu_y[dims])**2

        good = np.exp(-sqdiff_x / (2 * var_x[dims])) / np.sqrt(var_x[dims] * 2 * np.pi)
        bad = np.exp(-sqdiff_y / (2 * var_y[dims])) / np.sqrt(var_y[dims] * 2 * np.pi)

        good[:,0] *= alpha
        bad[:,0] *= (1-alpha)

        ratio = bad / (good + np.finfo(float).eps)
        logsum = np.log(ratio).sum(axis = -1)
        
        if normalize:
            logsum -= np.mean(logsum)

        # M Step
        if hard_em:
            z = logsum < 0
        else:
            z = 1.0/(1.0 + np.exp(logsum))
        
        keep = z > 1e-12
        assert keep.sum() != 0

        em_mapper.train(x[keep], y[keep], w=z[keep], **em_train_args)
        
        mu_y = np.average(y, axis=0, weights=1-z)
        var_y = np.average((y-mu_y)**2, axis=0, weights=1-z)
        
        mapped_x = em_mapper.apply(x[keep])
        avg_mapped_x = np.average(mapped_x, axis=0, weights=z[keep])
        var_x = np.average((mapped_x-avg_mapped_x)**2, axis=0, weights=z[keep])
        
        alpha = z.mean()
        
        argsort = logsum.argsort()
        logsum_sorted = logsum[argsort]
        z_sorted = z[argsort]
        pairs_sorted = [pairs[i] for i in argsort]
        
        if verbose:
            print("iter:", iteration, "alpha:", round(alpha, 4), "sample size:", sample_size, end = "     \r", flush = True)
        if verbose == 1:
            print()
            cutoff = (logsum < 0).sum()

            good_pairs_strings = [":".join(set(pair)) for pair in pairs_sorted[:cutoff]]
            bad_pairs_strings = [":".join(set(pair)) for pair in pairs_sorted[cutoff:]]
            
            print("good pairs:", " ".join(good_pairs_strings[:5] + ["..."] + good_pairs_strings[-5:]))
            print("bad pairs:", " ".join(bad_pairs_strings[:5] + ["..."] + bad_pairs_strings[-5:]))
            print()
    
    if verbose:
        print()

    mapper.train(x[keep], y[keep], w=z[keep], **train_args)
    return logsum_sorted, z_sorted, pairs_sorted


def test_mapper(src, tgt, pairs, mapper):
    from sklearn.metrics.pairwise import cosine_similarity
    
    topK = {1: 0.0, 5: 0.0, 10: 0.0}
    cosines = []
    
    x = src[[i for i,_ in pairs]]
    y = tgt[tgt.vocab]
    vocab2idx = {word: i for i, word in enumerate(tgt.vocab)}
    
    i2j = {i: set() for i,_ in pairs}
    for i,j in pairs:
        i2j[i].add(vocab2idx[j])#tgt.index(j))

    similarities = cosine_similarity(mapper.apply(x), y)
    for similarity, (i,j) in zip(similarities, pairs):
        for K in topK:
            top_partition = np.argpartition(similarity, kth=-K)[-K:]
            if any([j in i2j[i] for j in top_partition]):
                topK[K] += 1.0/similarities.shape[0]

        for j in i2j[i]:
            cosines.append(similarity[j])
    
    return np.mean(cosines), topK


def load_dico(path):
    if os.path.exists(f"{path}.pickle"):
        import _pickle
        with open(f"{path}.pickle", "rb") as handle:
            dico = _pickle.load(handle)

    elif os.path.exists(f"{path}.txt"):
        with open(f"{path}.txt") as handle:
            dico = [tuple(line.strip().split("\t")) for line in handle if len(line.strip())]
        if all([len(entry) == 1 for entry in dico]):
            dico = [entry + entry for entry in dico]
        assert all([len(entry) == 2 for entry in dico])
    
    else:
        raise Exception(f"Found neither {path}.pickle for {path}.txt")

    return dico


def parse_args():
    parser = argparse.ArgumentParser()
   
    transformer_choices = (\
            "bert-base-uncased", "bert-large-uncased",
            "bert-base-cased", "bert-large-cased", 
            "xlnet-base-cased", "xlnet-large-cased",
            "roberta-base", "roberta-large",
            "xlm-mlm-en-2048")

    #required arguments
    parser.add_argument("--src", type = str, required = True)
    parser.add_argument("--tgt", type = str, required = True)
    parser.add_argument("--dico", type = str, default = "")
    parser.add_argument("--test_dico", type = str, default = "")

    parser.add_argument("--verbose", type = int, default = 0)
    parser.add_argument("--seed", type = int, default = 0)

    parser.add_argument("--mlp_hidden_sizes", type = int, nargs = "+", default = [10000])
    parser.add_argument("--mlp_activation", choices = ("relu", "tanh"), default = "relu")
    parser.add_argument("--mlp_batch_size", type = int, default = 10000)
    parser.add_argument("--mlp_steps", type = int, default = 3000)

    parser.add_argument("--mlp_loss", default = "mse")
    parser.add_argument("--mlp_optimizer", default = "adam")

    parser.add_argument("--em_iterations", type = int, default = 100)
    parser.add_argument("--em_soft", action = "store_true")
    parser.add_argument("--em_sample_size", type = int, default = 200)
    parser.add_argument("--em_out", type = str, default = "", 
            help = "File where the output of the EM algorithm should be stored: \
                    A list of word pairs sorted by their probability of being 'good' pairs.")
    parser.add_argument("--save_out", type = str, default = "")

    parser.add_argument("--do_test", action = "store_true")

    parser.add_argument("--do_em", action = "store_true")
    parser.add_argument("--mapper", choices = ("linear", "mlp", "ortho"), default = "linear")
    parser.add_argument("--em_mapper", choices = ("linear", "mlp", "ortho"), default = "")

    parser.add_argument("--testratio", type = float, default = 0.1)

    args = parser.parse_args()
    
    if len(args.dico) == 0:
        args.dico = os.path.join(RESOURCE_DIR, f"{args.tgt}.all.dico")

    if len(args.em_mapper) == 0:
        args.em_mapper = args.mapper

    print(args)
    return args



if __name__ == "__main__":
    args = parse_args()

    outputstring = f"{args.src}:{args.tgt}:{args.mapper}"
    if args.do_em:
        outputstring += f"-em-{args.em_mapper}"
    
    train_dico = load_dico(args.dico)
    if args.do_test:
        if args.test_dico:
            test_dico = load_dico(args.test_dico)
        else:
            assert 0.0 < args.testratio < 1.0
            test_rate = int(1/args.testratio)
            test_dico = [train_dico[i] for i in range(len(train_dico)) if (i+1)%test_rate == 0]
            train_dico = [train_dico[i] for i in range(len(train_dico)) if (i+1)%test_rate != 0]
    else:
        test_dico = []
    
    print("*" * 50)
    print(outputstring, "Loading", args.tgt)
    tgt = load_embedding(args.tgt)
    print(outputstring, "Loading", args.src)
    src = load_embedding(args.src, do_lower_case = "uncased" in args.src)
    print("*" * 50)
    
    # we filter out words that are unknown to FastText (uncommon) or to the transformer (common)
    # we also filter out "special tokens" (FastText has none, transformer has a few)
    func = lambda ij: ij[0] in src and ij[1] in tgt and \
            (not ij[0] in src.all_special_tokens) and (not ij[1] in tgt.all_special_tokens)

    train_dico = list(filter(func, train_dico))
    test_dico = list(filter(func, test_dico))

    print("*"*50)
    print(outputstring, "Training on", len(train_dico), "words")
    print(outputstring, "Testing on", len(test_dico), "words")
    print("src dim", src[train_dico[0][0]].shape[-1])
    print("tgt dim", tgt[train_dico[0][1]].shape[-1])
    print("*"*50)

    all_mappers = {"linear": LinearMapper, "mlp": MLPMapper, "ortho": OrthogonalMapper}
    all_train_args = {"linear": {}, "ortho": {}, "mlp": {\
            "loss": args.mlp_loss, "optimizer": args.mlp_optimizer, "steps": args.mlp_steps,
            "hidden_sizes": args.mlp_hidden_sizes, "batchsize": args.mlp_batch_size,
            "activation": args.mlp_activation, "verbose": args.verbose, "seed": args.seed}}

    mapper, train_args = all_mappers[args.mapper](), all_train_args[args.mapper]

    if args.do_em:
        em_mapper, em_train_args = all_mappers[args.em_mapper](), all_train_args[args.em_mapper]
        logsum_sorted, z_sorted, pairs_sorted = train_mapper_em(src, tgt, train_dico, mapper=mapper, em_mapper=em_mapper, 
                train_args=train_args, em_train_args=em_train_args, seed=args.seed, hard_em = not args.em_soft,
                iterations=args.em_iterations, sample_size=args.em_sample_size, verbose=args.verbose)
        assert len(pairs_sorted) == len(train_dico) == len(logsum_sorted) == len(z_sorted)

        if len(args.em_out):
            lines = ["{}\t{}\t{:.5}\t{:.5}".format(i,j,float(p),l) for (i,j), p, l in \
                    zip(pairs_sorted, z_sorted, logsum_sorted)]
            with open(args.em_out, "w") as handle:
                handle.write("\n".join(lines))
    
    else:
        train_mapper(src, tgt, train_dico, mapper, train_args=train_args)
    
    print("Training done")
    print("*"*50)
    
    if args.do_test:
        train_cosine, train_topK = test_mapper(src, tgt, train_dico, mapper)
        train_topKstring = " / ".join(["K={}: {:.4}".format(k, train_topK[k]) for k in train_topK])
        print(outputstring, "*** Train results: Mean cosine {:.4} / TopK-precision {}".format(train_cosine, train_topKstring))
        test_cosine, test_topK = test_mapper(src, tgt, test_dico, mapper)
        test_topKstring = " / ".join(["K={}: {:.4}".format(k, test_topK[k]) for k in test_topK])
        print(outputstring, "*** Test results: Mean cosine {:.4} / TopK-precision {}".format(test_cosine, test_topKstring))     
        print("*"*50)

    if len(args.save_out):
        mapper.save(args.save_out)
