import os

from emb_input_transformers import EmbInputBertModel
from pytorch_transformers import BertForMaskedLM, RobertaForMaskedLM

from utils.util_wikidata import *
from embeddings import load_embedding, MappedEmbedding
from mappers import load_mapper

from tqdm import tqdm

import json
import numpy as np
import argparse
import torch

import pandas

def test(queries, method, model, language_model, 
        model_emb, wiki_emb, mapper, batch_size=4, 
        allowed_vocabulary=None):
    
    template = queries[0]["template"]
    relation = queries[0]["predicate_id"]
    
    assert all([query["template"] == template for query in queries])
    assert all([query["predicate_id"] == relation for query in queries])

    if allowed_vocabulary is None:
        restrict_vocabulary = []
    else:
        restrict_vocabulary = np.array([not i in allowed_vocabulary for i in range(model.config.vocab_size)])

    template = template.replace("[X]", model_emb.tokenizer.unk_token) 
    template = template.replace("[Y]", model_emb.tokenizer.mask_token) 

    template_encoded = model_emb.tokenizer.encode(template, add_special_tokens = True)
    template_tokenized = model_emb.tokenizer.convert_ids_to_tokens(template_encoded)

    template_vectors = [model_emb[token] for token in template_tokenized]
    unk_idx = template_tokenized.index(model_emb.tokenizer.unk_token)
    mask_idx = template_tokenized.index(model_emb.tokenizer.mask_token)
    
    if mask_idx > unk_idx:
        mask_idx -= len(template_tokenized)
    
    pad_vector = model_emb[model_emb.tokenizer.pad_token]
    slash_vector = model_emb["/"]

    batch = []
    template_fillers = []
    replaceable = []
    probs = []
    mean_attention = []

    mapped_titles = {}
    title_vectors = []


    for query in queries:
        replaceable.append(0)
        for title in sorted(list(query["wiki_titles"])):
            if title in wiki_emb:
                replaceable[-1] = 1
                if not title in mapped_titles:
                    mapped_titles[title] = len(mapped_titles)
                    title_vectors.append(wiki_emb[title])
                break
    
    assert len(mapped_titles) == len(title_vectors)

    if len(mapped_titles):
        title_vectors = np.array(title_vectors)
        mapped_title_vectors = mapper.apply(title_vectors)

    for i, query in enumerate(tqdm(queries, desc = method), start=1):
        subject_tokenized = model_emb.tokenizer.tokenize(query["sub_label"])
        vectors_tokenized = [model_emb[token] for token in subject_tokenized]
        vectors_wiki = vectors_slash = vectors_tokenized
        subject_wiki = subject_slash = subject_tokenized
        
        for title in query["wiki_titles"]:
            if title in mapped_titles:
                subject_wiki = [title]
                vectors_wiki = [mapped_title_vectors[mapped_titles[title]]]
                subject_slash = subject_wiki + ["/"] + subject_tokenized
                vectors_slash = vectors_wiki + [slash_vector] + vectors_tokenized

        batch.extend([vectors_tokenized, vectors_wiki, vectors_slash])
        template_fillers.extend([subject_tokenized, subject_wiki, subject_slash])

        if len(batch) >= batch_size or i == len(queries):
            maxlen = max([len(x) for x in batch]) + len(template_vectors) - 1
            input_vectors = torch.tensor([[pad_vector for _ in range(maxlen)] for _ in batch])
            attention_masks = torch.zeros((len(batch), maxlen))
            
            for batch_i, sample in enumerate(batch):
                template_filled = template_vectors[:unk_idx] + sample + template_vectors[unk_idx+1:]
                input_vectors[batch_i, :len(template_filled)] = torch.tensor(template_filled)
                attention_masks[batch_i, :len(template_filled)] = 1

            input_vectors = input_vectors.to(device = next(model.parameters()).device)
            attention_masks = attention_masks.to(device = next(model.parameters()).device)
                
            #hidden_states = model(input_ids = input_vectors, attention_mask = attention_masks)[0]
            tmp = model(input_ids = input_vectors, attention_mask = attention_masks)
            hidden_states = tmp[0]
            attentions = torch.stack(tmp[-1], 0).mean(0).mean(1).mean(1).detach().cpu().numpy()
            assert np.allclose(attentions.sum(-1), np.ones((attentions.shape[0],)))
            lm_inputs = torch.zeros_like(hidden_states[:,0])

            for batch_i, _ in enumerate(batch):
                lm_inputs[batch_i] = hidden_states[batch_i][attention_masks[batch_i] == 1][mask_idx]

            logits = language_model(lm_inputs)
            logits[:,restrict_vocabulary] = -10000 
            prob = logits.softmax(-1)
            prob = prob.detach().cpu().numpy()
            
            attn = [[float(a) for a in x] for x in attentions]
            assert len(attn) == prob.shape[0]

            mean_attention.extend(attn)
            probs.extend(prob)
            batch.clear()


    encoded = [model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False) for query in queries]
    assert all([len(x) == 1 for x in encoded])
    gold_answers = [x[0] for x in encoded]

    assert len(gold_answers) == len(replaceable) == len(queries)
    assert 3 * len(gold_answers) == len(template_fillers) == len(probs)
    assert len(probs) == len(mean_attention)

    for i, query in enumerate(queries):
        query_bert = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i] + template_tokenized[unk_idx+1:])
        query_wiki = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i+1] + template_tokenized[unk_idx+1:])
        query_slash = " ".join(template_tokenized[:unk_idx] + template_fillers[3*i+2] + template_tokenized[unk_idx+1:])

        assert (not "query_bert" in query) or query_bert == query["query_bert"]
        assert (not "query_wiki" in query) or query_wiki == query["query_wiki"]
        assert (not "query_slash" in query) or query_slash == query["query_slash"]
        
        query["query_bert"] = query_bert
        query["query_wiki"] = query_wiki
        query["query_slash"] = query_slash

        prob = {}
        prob["bert"] = probs[3*i]
        prob[method + "_replace"] = probs[3*i+1]
        prob[method + "_concat"] = probs[3*i+2]
        
        prob[method + "_ens_avg"] = (prob["bert"] + prob[method + "_replace"]) / 2
        prob[method + "_ens_max"] = prob["bert"] if prob["bert"].max() > prob[method + "_replace"].max() else prob[method + "_replace"]

        query["attn:bert"] = mean_attention[3*i][:len(query_bert)]
        query["attn:" + method + "_replace"] = mean_attention[3*i+1][:len(query_wiki)]
        query["attn:" + method + "_concat"] = mean_attention[3*i+2][:len(query_slash)]
        
        for key in prob:
            ranking = np.argsort(prob[key])[::-1]
            top10 = tuple(model_emb.tokenizer.convert_ids_to_tokens(ranking[:10]))
            top10_prob = tuple([float(p) for p in prob[key][ranking[:10]]])

            assert (not f"top10_prob:{key}" in query) or query[f"top10_prob:{key}"] == top10_prob
            assert (not f"top10:{key}" in query) or query[f"top10:{key}"] == top10
            query[f"top10:{key}"] = top10
            query[f"top10_prob:{key}"] = top10_prob
            
            assert (not "replaceable" in query) or query["replaceable"] == replaceable[i]
            query["replaceable"] = replaceable[i]

            gold_rank = int(np.where(ranking == gold_answers[i])[0][0])+1
            assert (not f"gold_rank:{key}" in query) or query[f"gold_rank:{key}"] == gold_rank
            query[f"gold_rank:{key}"] = gold_rank
            
            gold_prob = round(float(prob[key][gold_answers[i]]), 5)
            assert (not f"gold_prob:{key}" in query) or query[f"gold_prob:{key}"] == gold_prob
            query[f"gold_prob:{key}"] = gold_prob



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type = str, default = "../data/LAMA/data")
    parser.add_argument("--methods", type = str, nargs = "+", default = ("linear",))
    parser.add_argument("--modelname", type = str, default = "bert-base-cased")
    parser.add_argument("--wikiname", type = str, default = "wikipedia2vec-base-cased")
    parser.add_argument("--device", type = str, default = "cuda")
    parser.add_argument("--uhn", action = "store_true", default = False)
    parser.add_argument("--infer_entity", action = "store_true", default = False)
    parser.add_argument("--allowed_vocabulary", type = str, default = "../resources/common_vocab_cased.txt")
    parser.add_argument("--output_dir", type = str, default = "../outputs/LAMA")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    patterns = [\
            {"relation": "place_of_birth", "template": "[X] was born in [Y] ."},
            {"relation": "date_of_birth", "template": "[X] (born [Y])."},
            {"relation": "place_of_death", "template": "[X] died in [Y] ."}]
    
    with open(os.path.join(args.data_dir, "relations.jsonl")) as handle:
        patterns.extend([json.loads(line) for line in handle])

    all_queries = []

    wiki_emb = load_embedding(args.wikiname)
    model_emb = load_embedding(args.modelname)

    allowed_vocabulary = None
    if args.allowed_vocabulary:
        with open(args.allowed_vocabulary) as handle:
            lines = [line.strip() for line in handle]
        encoded = [model_emb.tokenizer.encode(token, add_special_tokens = False) for token in lines]
        assert all([len(x) == 1 for x in encoded])
        allowed_vocabulary = set([x[0] for x in encoded if len(x) == 1])

    model = EmbInputBertModel.from_pretrained(args.modelname, output_attentions = True)
    language_model = BertForMaskedLM.from_pretrained(args.modelname).cls

    model = model.to(device = args.device)
    language_model = language_model.to(device = args.device)

    model.eval()
    language_model.eval()
    
    mappers = {method: load_mapper(f"{args.wikiname}.{args.modelname}.{method}") for method in args.methods}
    
    for pattern in tqdm(patterns):
        relation, template = pattern["relation"], pattern["template"]
        
        uhn_suffix = "_UHN" if args.uhn else ""
        
        if relation.startswith("P"):
            path = os.path.join(args.data_dir, f"TREx{uhn_suffix}/{relation}.jsonl")
        else:
            path = os.path.join(args.data_dir, f"Google_RE{uhn_suffix}/{relation}_test.jsonl")
        
        if not os.path.exists(path):
            continue
        
        with open(path) as handle:
            queries = [json.loads(line) for line in handle]        

        if args.infer_entity:
            label2uri = label2wikidata([query["sub_label"] for query in queries], verbose = False)
            all_uris = set()
            for label in label2uri:
                uris = list(label2uri[label])
                label2uri[label] = sorted(list(label2uri[label]), key = lambda x:int(x[1:]))
                all_uris.update(uris)

            uri2title = wikidata2title(all_uris, verbose = False)
            label2title = {}

            for label in label2uri:
                label2title[label] = []
                for uri in label2uri[label]:
                    label2title[label].extend(sorted(list(uri2title[uri])))

            for query in queries:
                query["wiki_titles"] = ["ENTITY/" + x for x in label2title[query["sub_label"]]]

        assert all([len(model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)) == 1 for query in queries])

        if "uncased" in args.modelname:
            for query in queries:
                obj_label = model_emb.tokenizer.tokenize(query["obj_label"])
                assert len(obj_label) == 1
                query["obj_label"] = obj_label[0]

        if allowed_vocabulary:
            for query in queries:
                if not model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)[0] in allowed_vocabulary:
                    print(query["obj_label"], "not in allowed vocab")
            queries = [query for query in queries if \
                    model_emb.tokenizer.encode(query["obj_label"], add_special_tokens = False)[0] in allowed_vocabulary]
        
        if not relation.startswith("P"):
            for query in queries:
                query["predicate_id"] = query["pred"].split("/")[-1]
        

        if not args.infer_entity:
            if relation.startswith("P"):
                if "wikipedia2vec" in args.wikiname:
                    mapping = wikidata2title([query["sub_uri"] for query in queries], verbose = False)
                    for query in queries:
                        query["wiki_titles"] = ["ENTITY/" + x for x in mapping[query["sub_uri"]]]
                else:
                    for query in queries:
                        query["wiki_titles"] = [query["sub_uri"]]

            else:
                all_wikidata = [query["sub_w"] for query in queries if query["sub_w"]]
                all_titles = []
                for query in queries:
                    all_titles.extend([title_url2title(evidence["url"]) for evidence in query["evidences"]])

                if "wikipedia2vec" in args.wikiname:
                    mapping = wikidata2title(all_wikidata)
                else:
                    mapping = title2wikidata(all_titles)

                for query in queries:
                    query["wiki_titles"] = set()

                    if query["sub_w"]:
                        if "wikipedia2vec" in args.wikiname:
                            query["wiki_titles"].update(["ENTITY/" + x for x in mapping[query["sub_w"]]])
                        else:
                            query["wiki_titles"].add(query["sub_w"])

                    for title in [title_url2title(evidence["url"]) for evidence in query["evidences"]]:
                        if "wikipedia2vec" in args.wikiname:
                            query["wiki_titles"].add("ENTITY/" + title)
                        else:
                            query["wiki_titles"].update(mapping[title])
                
                    query["wiki_titles"] = sorted(list(query["wiki_titles"]))


        for query in queries:
            query["template"] = template
            if "evidences" in query:
                del query["evidences"]
            if "judgments" in query:
                del query["judgments"]
            if "masked_sentence" in query:
                del query["masked_sentence"]

        for method in tqdm(args.methods, desc = relation):
            test(queries, method, model, language_model, 
                    model_emb = model_emb, wiki_emb = wiki_emb, 
                    mapper = mappers[method],
                    allowed_vocabulary = allowed_vocabulary)

        inf = "_infer" if args.infer_entity else ""
    
        with open(os.path.join(args.output_dir, f"{relation}.{args.modelname}.{args.wikiname}{inf}.jsonl"), "w") as handle:
            handle.write("\n".join([json.dumps(query) for query in queries]))
    
        all_queries.extend(queries)
    
    with open(os.path.join(args.output_dir, f"all.{args.modelname}.{args.wikiname}{inf}.jsonl"), "w") as handle:
        handle.write("\n".join([json.dumps(query) for query in all_queries]))
        
