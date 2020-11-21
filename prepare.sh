if [[ ! -d code/kb ]]; then
	cd code
	git clone https://github.com/allenai/kb
	cd kb
	pip install -r requirements.txt
	python -c "import nltk; nltk.download('wordnet')"
	python -m spacy download en_core_web_sm
	pip install --editable .
	cd ../..
fi


for f in aida_train.txt aida_dev.txt aida_test.txt; do
	if [[ ! -f data/AIDA/$f ]]; then
		wget https://allennlp.s3-us-west-2.amazonaws.com/knowbert/wiki_entity_linking/$f -O data/AIDA/$f
	fi
done

if [[ ! -d LAMA ]]; then
	git clone https://github.com/facebookresearch/LAMA
	cd LAMA
	python3 setup.py install
	cd ..
fi

if [[ ! -d data/LAMA ]]; then
	mkdir data/LAMA
fi

if [[ ! -d data/LAMA/data ]]; then
	wget https://dl.fbaipublicfiles.com/LAMA/data.zip -O data/LAMA/data.zip
	cd data/LAMA
	unzip data
	cd ../..
fi

for d in TREx Google_RE; do
	if [[ ! -d data/LAMA/data/${d}_UHN ]]; then
		python3 LAMA/scripts/create_lama_uhn.py --srcdir data/LAMA/data/$d
	fi
done;
