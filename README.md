```bash
conda env create -f environment.yaml
conda activate e-bert
```

Download and prepare necessary data for the experiments
```bash
bash prepare.sh
```

If you want to train your own entity embeddings, run Wikipedia2Vec. This will take a while:
```bash
cd resources/wikipedia2vec
wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2

wikipedia2vec build-dump-db enwiki-latest-pages-articles.xml.bz2 wikipedia2vec.db
wikipedia2vec build-dictionary --no-lowercase wikipedia2vec.db wikipedia2vec-cased-raw_dic.pkl
python3 extend_entity_dict.py --src wikipedia2vec-cased-raw_dic.pkl --tgt wikipedia2vec-cased_dic.pkl --dumpdb wikipedia2vec.db --entity_file ../../data/AIDA/aida_entities_and_candidates.txt

wikipedia2vec build-link-graph wikipedia2vec.db wikipedia2vec-cased_dic.pkl wikipedia2vec-cased_lg.pkl
wikipedia2vec build-mention-db wikipedia2vec.db wikipedia2vec-cased_dic.pkl wikipedia2vec-cased_mention.pkl
time wikipedia2vec train-embedding --dim-size 768 --mention-db wikipedia2vec-cased_mention.pkl --link-graph wikipedia2vec-cased_lg.pkl wikipedia2vec.db wikipedia2vec-cased_dic.pkl wikipedia2vec-base-cased


cd ../..
```

Or download our pretrained model:
```bash
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased
wget https://www.cis.uni-muenchen.de/~poerner/blobs/e-bert/wikipedia2vec-base-cased.bert-base-cased.linear.npy
mv wikipedia2vec-base-cased resources/wikipedia2vec
mv wikipedia2vec-base-cased.bert-base-cased.linear.npy mappers
```

Fit the linear mapping:
```bash
cd code
python3 run_mapping.py --src wikipedia2vec-base-cased --tgt bert-base-cased --save_out ../mappers/wikipedia2vec-base-cased.bert-base-cased.linear
```

Run LAMA experiment:
```bash
python3 run_lama.py --wikiname wikipedia2vec-base-cased --modelname bert-base-cased --data_dir ../data/LAMA/data --output_dir ../outputs/LAMA --infer_entity
python3 score_lama.py --file ../outputs/LAMA/all.bert-base-cased.wikipedia2vec-base-cased_infer.jsonl
```

Run LAMA-UHN experiment:
```bash
python3 run_lama.py --wikiname wikipedia2vec-base-cased --modelname bert-base-cased --data_dir ../data/LAMA/data --output_dir ../outputs/LAMA_UHN --infer_entity --uhn
python3 score_lama.py --file ../outputs/LAMA_UHN/all.bert-base-cased.wikipedia2vec-base-cased_infer.jsonl
```


Download and unpack RC (FewRel) data from `https://drive.google.com/open?id=1HlWw7Q6-dFSm9jNSCh4VaBf1PlGqt9im` and unpack them in data/fewrel

Run Relation Classification (FewRel) experiment
```bash
python3 run_fewrel.py --data_dir ../data/fewrel --output_dir ../outputs/fewrel/ebert_concat --ent concat --do_train --do_eval --do_test
python3 score_fewrel.py ../outputs/fewrel/ebert_concat/test_gold.txt ../outputs/fewrel/ebert_concat/test_pred.txt
```
