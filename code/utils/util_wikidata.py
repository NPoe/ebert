import requests
import json
import re
import urllib.parse
from tqdm import tqdm
from collections import Counter

HEADERS = {"User-agent": "CIS Munich"}
SPARQL_API = "https://query.wikidata.org/sparql"
WIKIDATA_RGX = re.compile("\AQ[0-9]+\Z")
WIKI_API = "https://en.wikipedia.org/w/api.php"


def wikidata_url2id(wikidata_url):
    wikidata_id = wikidata_url.split("/")[-1]
    assert WIKIDATA_RGX.match(wikidata_id)
    return wikidata_id

def title_url2title(title_url):
    title = title_url.split("/")[-1]
    return urllib.parse.unquote(title)

def query_sparql(query):
    params = {"query": query, "format": "json"}
    ret = requests.get(SPARQL_API, params = params, headers = HEADERS)
    return json.loads(ret.content)

def query_wiki(titles):
    params = {"action": "query", "prop": "pageprops", "ppprop": "wikibase_item",
            "redirects": "10", "titles": "|".join(titles), "format": "json"}
    ret = requests.get(WIKI_API, params = params, headers = HEADERS)
    return json.loads(ret.content)

def batchify(data, batch_size):
    return [data[i:i+batch_size] for i in range(0, len(data), batch_size)]

def string2list(string_or_list):
    if isinstance(string_or_list, str):
        return [string_or_list]
    return sorted(list(set(string_or_list)))

def init_mapping(data):
    return {item: set() for item in data}

def try_batch(func, batch, raise_exceptions):
    mapping = init_mapping(batch)
    
    try:
        mapping.update(func(batch))
    
    except Exception as e:
        if len(batch) == 1:
            print("Final fail", func, batch[0])
            print(e)
            if raise_exceptions:
                raise e

        else:
            new_size = max(1, len(batch)//5)
            for sub_batch in batchify(batch, new_size):
                mapping.update(try_batch(func, sub_batch, raise_exceptions))

    return mapping

def try_batches(func, data, batch_size, verbose, raise_exceptions = False, msg = None):
    data = string2list(data) 
    mapping = init_mapping(data)
    batches = batchify(data, batch_size)

    if verbose:
        batches = tqdm(batches, desc = msg)

    for batch in batches:
        mapping.update(try_batch(func, batch, raise_exceptions))
    
    return mapping



def freebase2wikidata_inner(freebase_ids):
    mapping = init_mapping(freebase_ids)

    joined_string = " ".join(["'" + i + "'" for i in freebase_ids])
    query = "SELECT ?wikidata ?freebase WHERE { \n \
            VALUES ?freebase { " + joined_string + " } .\n \
            ?wikidata wdt:P646 ?freebase . }"
   
    result = query_sparql(query)
    for binding in result["results"]["bindings"]:
        wikidata_id = wikidata_url2id(binding["wikidata"]["value"])
        mapping[binding["freebase"]["value"]].add(wikidata_id)

    return mapping

def wikidata2freebase_inner(wikidata_ids):
    mapping = init_mapping(wikidata_ids)

    joined_string = " ".join([f"wd:{i}" for i in wikidata_ids])
    query = "SELECT ?wikidata ?freebase WHERE { \n \
            VALUES ?wikidata { " + joined_string + " } .\n \
            ?wikidata wdt:P646 ?freebase . }"
  
    result = query_sparql(query)
    for binding in result["results"]["bindings"]:
        wikidata_id = wikidata_url2id(binding["wikidata"]["value"])
        mapping[wikidata_id].add(binding["freebase"]["value"])

    return mapping 

def wikidata2rel_inner(wikidata_ids):
    mapping = init_mapping(wikidata_ids)

    joined_string = " ".join([f"wd:{i}" for i in wikidata_ids])
    query = "SELECT ?wikidata ?rel WHERE { \n \
            VALUES ?wikidata { wd:" + joined_string + " } .\n \
            {?wikidata ?p ?rel} . \n \
            ?p rdf:type ?ptype . \n \
            VALUES ?ptype { owl:ObjectProperty } . \n \
            FILTER (!REGEX(STR(?rel), \"-*/entity/statement/.*\")) \n \
            FILTER (REGEX(STR(?rel), \".*/entity/Q[0-9]+\")) }"
            
    result = query_sparql(query)
    for binding in result["results"]["bindings"]:
        wikidata_id = wikidata_url2id(binding["wikidata"]["value"])
        rel_id = wikidata_url2id(binding["rel"]["value"])
        mapping[wikidata_id].add(rel_id)

    return mapping 


def wikidata2title_inner(wikidata_ids):
    mapping = init_mapping(wikidata_ids)
    
    joined_string = " ".join([f"wd:{i}" for i in wikidata_ids])
    query = "SELECT ?wikidata ?title WHERE {\n \
            VALUES ?wikidata { " + joined_string + " } .\n \
            ?title schema:about ?wikidata . \n \
            ?title schema:inLanguage 'en' .\n \
            FILTER REGEX(str(?title), '.*en.wikipedia.org.*') .}"

    result = query_sparql(query)
    for binding in result["results"]["bindings"]:
        wikidata_id = wikidata_url2id(binding["wikidata"]["value"])
        title = title_url2title(binding["title"]["value"])
        mapping[wikidata_id].add(title)

    return mapping

def label2wikidata_inner(labels):
    mapping = init_mapping(labels)
    
    labels = [label.replace("'", "\\'") for label in labels]
    joined_string = " ".join([f"'{i}'@en" for i in labels])
    query = "SELECT ?wikidata ?label WHERE {\n \
        ?wikidata rdfs:label ?label . \n \
        VALUES ?label { " + joined_string + " }\n \
        FILTER((LANG(?label)) = 'en') . }"

    result = query_sparql(query)
    for binding in result["results"]["bindings"]:
        if not WIKIDATA_RGX.match(binding["wikidata"]["value"].split("/")[-1]):
            continue
        wikidata_id = wikidata_url2id(binding["wikidata"]["value"])
        label = binding["label"]["value"]
        mapping[label].add(wikidata_id)
    return mapping


def redirect_titles_inner(titles):
    mapping = {}

    result = query_wiki(titles)
    normalized = result["query"].get("normalized", [])
    redirects = result["query"].get("redirects", [])
        
    normalized = set([(x["from"], x["to"]) for x in normalized if x["to"] != x["from"]])
    redirects = set([(x["from"], x["to"]) for x in redirects if x["to"] != x["from"]])
    assert len(normalized) == len(dict(normalized))
    assert len(redirects) == len(dict(redirects))
    
    redirects = dict(redirects)
    normalized = dict(normalized)
        
    pages = []
    for key in ("pages", "interwiki"):
        if key in result["query"]:
            if isinstance(result["query"][key], dict):
                pages.extend(result["query"][key].values())
            else:
                pages.extend(result["query"][key])

    norm_title2page = {page["title"]: page for page in pages}
       
    for title in titles:
        norm_title = title
        while norm_title in normalized:
            norm_title = normalized[norm_title]
        while norm_title in redirects:
            norm_title = redirects[norm_title]
        mapping[title] = "_".join(norm_title.split())

    return mapping


def title2wikidata_inner(titles):
    mapping = init_mapping(titles)

    result = query_wiki(titles)
    normalized = result["query"].get("normalized", [])
    redirects = result["query"].get("redirects", [])
        
    normalized = set([(x["from"], x["to"]) for x in normalized if x["to"] != x["from"]])
    redirects = set([(x["from"], x["to"]) for x in redirects if x["to"] != x["from"]])
    assert len(normalized) == len(dict(normalized))
    assert len(redirects) == len(dict(redirects))
    
    redirects = dict(redirects)
    normalized = dict(normalized)
        
    pages = []
    for key in ("pages", "interwiki"):
        if key in result["query"]:
            if isinstance(result["query"][key], dict):
                pages.extend(result["query"][key].values())
            else:
                pages.extend(result["query"][key])

    norm_title2page = {page["title"]: page for page in pages}
       
    for title in titles:
        norm_title = title
        while norm_title in normalized:
            norm_title = normalized[norm_title]
        while norm_title in redirects:
            norm_title = redirects[norm_title]
            
        page = norm_title2page[norm_title]
        if "pageprops" in page and "wikibase_item" in page["pageprops"]:
            mapping[title].add(page["pageprops"]["wikibase_item"])
    
    return mapping



def label2wikidata(labels, verbose = True):
    return try_batches(label2wikidata_inner, labels,
            batch_size = 200, verbose = verbose)

def title2wikidata(titles, verbose = True):
    return try_batches(title2wikidata_inner, titles,
            batch_size = 45, verbose = verbose)

def redirect_titles(titles, verbose = True, raise_exceptions = False, msg = None):
    return try_batches(redirect_titles_inner, titles,
            batch_size = 45, verbose = verbose, 
            raise_exceptions = raise_exceptions, msg = msg)

def freebase2wikidata(freebase_ids, verbose = True):
    return try_batches(freebase2wikidata_inner, freebase_ids, 
            batch_size = 200, verbose = verbose)

def wikidata2freebase(wikidata_ids, verbose = True):
    return try_batches(wikidata2freebase_inner, wikidata_ids, 
            batch_size = 200, verbose = verbose)

def wikidata2title(wikidata_ids, verbose = True):
    return try_batches(wikidata2title_inner, wikidata_ids, 
            batch_size = 200, verbose = verbose)

def wikidata2rel(wikidata_ids, verbose = True):
    return try_batches(wikidata2rel_inner, wikidata_ids, 
            batch_size = 200, verbose = verbose)

def pivot(func1, func2, data, verbose):
    mapping = init_mapping(data)
    mapping1 = func1(data, verbose = verbose)

    all_pivots = set()
    for pivots in mapping1.values():
        all_pivots.update(pivots)
    all_pivots = list(all_pivots)

    mapping2 = func2(all_pivots, verbose = verbose)
    for key in mapping:
        for pivot in mapping1[key]:
            mapping[key].update(mapping2[pivot])
    return mapping

def freebase2title(freebase_ids, verbose = True):
    return pivot(freebase2wikidata, wikidata2title, freebase_ids,
            verbose = verbose)

def title2freebase(titles, verbose = True):
    return pivot(title2wikidata, wikidata2freebase, titles, verbose = verbose)


def rename_wikipedia2vec_entities(src, tgt_w, tgt_f):
    from gensim.models import KeyedVectors
    import numpy as np

    old_model = KeyedVectors.load(src, mmap='r')
    words = [word for word in old_model.vocab.keys() if not word.startswith("ENTITY/")]

    titles = [word[7:] for word in old_model.vocab.keys() if word.startswith("ENTITY/")]
    titles = [title for title in titles if not "#" in title]
    title2deviant_title = {title.split("|")[0]: title for title in titles}
    titles = [title.split("|")[0] for title in titles]

    t2w = title2wikidata(titles) 
    t2f = init_mapping(titles)
    w2f = wikidata2freebase(sum([list(x) for x in t2w.values()], []))
   
    for title in t2w:
        for w in t2w[title]:
            for f in w2f[w]:
                t2f[title].add(f)

    w2t = init_mapping(sum([list(x) for x in t2w.values()], []))
    f2t = init_mapping(sum([list(x) for x in t2f.values()], []))

    for title in titles:
        for w in t2w[title]:
            w2t[w].add(title)
        for f in t2f[title]:
            f2t[f].add(title)

    print("Some stats", flush = True) 
    print("t2f", len(t2f), Counter([len(x) for x in t2f.values()]))
    print("w2f", len(w2f), Counter([len(x) for x in w2f.values()]))
    print("t2w", len(t2w), Counter([len(x) for x in t2w.values()]))
    print("f2t", len(f2t), Counter([len(x) for x in f2t.values()]))
    print("w2t", len(w2t), Counter([len(x) for x in w2t.values()]))

    w_vecs = {word: old_model[word] for word in words}
    freebase_model = KeyedVectors(old_model.vector_size)
    freebase_model.add(words, [w_vecs[word] for word in words])
    freebase_words = list(f2t.keys())
    f_vecs = {f: np.mean([old_model["ENTITY/" + title2deviant_title[title]] for title in f2t[f]], 0) for f in freebase_words}
    freebase_model.add(freebase_words, [f_vecs[word] for word in freebase_words])
    freebase_model.save(tgt_f)
    del freebase_model

    wikidata_model = KeyedVectors(old_model.vector_size)
    wikidata_model.add(words, [w_vecs[word] for word in words])
    wikidata_words = list(w2t.keys())
    w_vecs = {w: np.mean([old_model["ENTITY/" + title2deviant_title[title]] for title in w2t[w]], 0) for w in wikidata_words}
    wikidata_model.add(wikidata_words, [w_vecs[word] for word in wikidata_words]) 
    wikidata_model.save(tgt_w)
    del wikidata_model


if __name__ == "__main__":
    pass
    #print(label2wikidata(["Barack Obama", "Andean Baroque"]))
    ##import sys
    #src = sys.argv[-1]
    #fsrc = "../../resources/{}.gensim".format(src)
    #ftgt_w = "../../resources/{}.wikidata.gensim".format(src)
    #ftgt_f = "../../resources/{}.freebase.gensim".format(src)
    #rename_wikipedia2vec_entities(src=fsrc, tgt_w=ftgt_w, tgt_f=ftgt_f)



