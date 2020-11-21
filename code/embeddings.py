import numpy as np
import re
import gc

from config import *

def load_embedding(name, **kwargs): 
    if name.startswith("bert-") or name.startswith("xlnet-") or name.startswith("t5-") \
            or name.startswith("roberta-") or name.startswith("xlm-"):
        return TransformerEmbedding(name)
    if "wikipedia2vec" in name:
        kwargs = {key: kwargs[key] for key in kwargs if key in ("do_lower_case", "prefix")}
        return Wikipedia2VecEmbedding(name, **kwargs)
    raise Exception(f"Unknown name: {name}")


class Embedding:
    def __getitem__(self, word_or_words):
        if isinstance(word_or_words, str):
            if not word_or_words in self:
                raise Exception("Embedding does not contain", word_or_words)
            return self.getvector(word_or_words)
        
        for word in word_or_words:
            if not word in self:
                raise Exception("Embedding does not contain", word)
        
        return self.getvectors(word_or_words)
    
    @property
    def vocab(self):
        return self.get_vocab()

    @property
    def all_embeddings(self):
        return self[self.vocab]
   

class MappedEmbedding(Embedding):
    def __init__(self, embedding, mapper):
        self.embedding = embedding
        self.mapper = mapper

    def __getitem__(self, word_or_words):
        embedding = self.embedding[word_or_words]
        return self.mapper.apply(embedding)

    def __contains__(self, word):
        return word in self.embedding
    
    def index(self, word):
        return self.embedding.index(word)

    @property
    def all_special_tokens(self):
        return self.embedding.all_special_tokens
    


class GensimEmbedding(Embedding):
    def __init__(self, prefix = "", do_lower_case = False,
            start_chars = (ROBERTA_START_CHAR, XLNET_START_CHAR)):
        self.start_char_pat = re.compile("|".join(["\A" + char for char in start_chars]))
        self.do_lower_case = do_lower_case
        self.prefix = prefix

    def _preprocess_word(self, word):
        word = self.start_char_pat.sub("", word.strip())
        if self.do_lower_case:
            word = word.lower()
        return self.prefix + word

    def __contains__(self, word):
        word = self._preprocess_word(word)
        return word in self.vectors

    def getvector(self, word):
        word = self._preprocess_word(word)
        return self.vectors[word]
    
    @property
    def all_special_tokens(self):
        return []

    
    def getvectors(self, words):
        return np.stack([self.getvector(word) for word in words], 0)
    
    def get_vocab(self):
        return list(self.vectors.vocab)

    def index(self, word):
        word = self._preprocess_word(word)
        return self.vectors.vocab[word].index
    

class ConcEmbedding(Embedding):
    def __init__(self, embeddings):
        super(ConcEmbedding, self).__init__()
        self.embeddings = embeddings

    def getvector(self, word):
        return self.getvectors([word])[0]
    
    def getvectors(self, words):
        vectors = [emb.getvectors(words) for emb in self.embeddings]
        return np.concatenate(vectors, -1)

    def __contains__(self, word):
        return all([word in emb for emb in self.embeddings])
    
    
    @property
    def all_special_tokens(self):
        special_tokens = set()
        for emb in self.embeddings:
            special_tokens.update(emb.all_special_tokens)
        return sorted(list(special_tokens))

class KeyedVectorsEmbedding(GensimEmbedding):
    def __init__(self, path, prefix="", do_lower_case = False,
            start_chars = (ROBERTA_START_CHAR, XLNET_START_CHAR)):
        super(KeyedVectorsEmbedding, self).__init__(prefix=prefix, start_chars=start_chars, do_lower_case=do_lower_case)
        self._try_load(path)
        if hasattr(self.vectors, "wv"):
            self.vectors = self.vectors.wv
        self.embeddings = self.vectors.vectors


    def _try_load(self, path):
        if os.path.exists(path) and not os.path.isdir(path):
            if path.endswith(".gensim"):
                self._load_gensim(path)
            elif path.endswith(".txt") or path.endswith(".vec") or path.endswith(".bin"):
                self._load_txt(path)
            elif path.endswith(".model"):
                self._load_model(path)
        elif os.path.exists(f"{path}.gensim"):
            self._load_gensim(f"{path}.gensim")
        elif os.path.exists(f"{path}.txt"):
            self._load_txt(f"{path}.txt")
        elif os.path.exists(f"{path}.vec"):
            self._load_txt(f"{path}.vec")
        elif not path.startswith(RESOURCE_DIR):
            self._try_load(os.path.join(RESOURCE_DIR, path))
        else:
            raise Exception()

    def _load_txt(self, path):
        from gensim.models import KeyedVectors
        self.vectors = KeyedVectors.load_word2vec_format(path)

    def _load_gensim(self, path):
        from gensim.models import KeyedVectors
        self.vectors = KeyedVectors.load(path, mmap='r')
    
    def _load_model(self, path):
        from gensim.models import Word2Vec
        self.vectors = Word2Vec.load(path, mmap='r')

class Wikipedia2VecEmbedding(Embedding):
    def __init__(self, path, prefix = "ENTITY/", do_cache_dict = True, do_lower_case = False):
        from wikipedia2vec import Wikipedia2Vec, Dictionary
        if os.path.exists(path):
            self.model = Wikipedia2Vec.load(path)
        elif os.path.exists(os.path.join(RESOURCE_DIR, "wikipedia2vec", path)):
            self.model = Wikipedia2Vec.load(os.path.join(RESOURCE_DIR, "wikipedia2vec", path))
        else:
            raise Exception()

        self.dict_cache = None
        if do_cache_dict:
            self.dict_cache = {}

        self.prefix = prefix
        self.do_lower_case = do_lower_case

        assert self.prefix + "San_Francisco" in self
        assert self.prefix + "St_Linus" in self

    def _preprocess_word(self, word):
        if word.startswith(self.prefix):
            word = " ".join(word[len(self.prefix):].split("_"))
        if self.do_lower_case:
            word = word.lower()
        return word
    
    def index(self, word):
        prepr_word = self._preprocess_word(word)

        if (not self.dict_cache is None) and prepr_word in self.dict_cache:
            return self.dict_cache[prepr_word]

        if word.startswith(self.prefix):
            ret = self.model.dictionary.get_entity(prepr_word)
        else:
            ret = self.model.dictionary.get_word(prepr_word)

        if not self.dict_cache is None:
            self.dict_cache[prepr_word] = ret
        
        return ret

    def __contains__(self, word):  
        return self.index(word) is not None

    def getvector(self, word):
        if word.startswith(self.prefix):
            return self.model.get_vector(self.index(word))
        return self.model.get_vector(self.index(word))
    
    @property
    def all_special_tokens(self):
        return []

    def getvectors(self, words):
        return np.stack([self.getvector(word) for word in words], 0)



class FastTextEmbedding(GensimEmbedding):
    def __init__(self, path, prefix="", do_lower_case=False, 
            start_chars = (ROBERTA_START_CHAR, XLNET_START_CHAR)):
        super(FastTextEmbedding, self).__init__(prefix=prefix, start_chars=start_chars, do_lower_case=do_lower_case)
        self._try_load(path)
        self.embeddings = self.vectors.vectors

    def _try_load(self, path):
        if os.path.exists(path):
            if path.endswith(".gensim") or path.endswith(".model"):
                self._load_gensim(path)
            elif path.endswith(".bin"):
                self._load_bin(path)
        elif os.path.exists(f"{path}.gensim"):
            self._load_gensim(f"{path}.gensim")
        elif os.path.exists(f"{path}.bin"):
            self._load_bin(f"{path}.bin")
        elif not path.startswith(RESOURCE_DIR):
            self._try_load(os.path.join(RESOURCE_DIR, path))

        if not hasattr(self, "vectors"):
            raise Exception("Loading failed", path)

    def _load_bin(self, path):
        from gensim.models.fasttext import load_facebook_vectors
        self.vectors = load_facebook_vectors(path)
    
    def _load_gensim(self, path):
        from gensim.models import FastText
        self.vectors = FastText.load(path, mmap='r').wv

class CombinedEmbedding(Embedding):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    @property
    def tokenizer(self):
        tokenizer = None
        for embedding in self.embeddings:
            if hasattr(embedding, "tokenizer"):
                if not tokenizer is None:
                    raise Exception()
                tokenizer = embedding.tokenizer
        return tokenizer

    def get_vocab(self):
        vocab = []
        for embedding in self.embeddings:
            vocab.extend(embedding.get_vocab())
       
        if len(vocab) != len(set(vocab)):
            raise Exception()

        return vocab
    

    def _get_embedding_that_contains(self, word):
        ret = None
        for i, embedding in enumerate(self.embeddings):
            if word in embedding:
                if not ret is None:
                    raise Exception()
                ret = embedding

        return ret

    def __contains__(self, word):
        embedding = self._get_embedding_that_contains(word)
        return embedding is not None

    def getvector(self, word):
        embedding = self._get_embedding_that_contains(word)
        return embedding.getvector(word)

    def getvectors(self, words):
        vectors = []
        for word in words:
            vectors.append(self.getvector(word))
        return np.array(vectors)

    @property
    def all_special_tokens(self):
        special_tokens = []
        for embedding in self.embeddings:
            special_tokens.extend(embedding.all_special_tokens)
        return special_tokens

class TransformerEmbedding(Embedding):
    @staticmethod
    def get_model_and_tokenizer_class(name):
        modeltype = name.split("-")[0].split("/")[-1]

        if modeltype == "bert" or "pretraining" in name:
            from pytorch_transformers import BertModel, BertTokenizer
            return BertModel, BertTokenizer
        elif modeltype == "xlnet":
            from pytorch_transformers import XLNetModel, XLNetTokenizer
            return XLNetModel, XLNetTokenizer
        elif modeltype == "roberta":
            from pytorch_transformers import RobertaModel, RobertaTokenizer
            return RobertaModel, RobertaTokenizer
        elif modeltype == "xlm":
            from pytorch_transformers import XLMModel, XLMTokenizer
            return XLMModel, XLMTokenizer, EmbInputXLMTokenizer
        else:
            raise Exception(f"Unknown model type: {name}")

    def get_vocab(self):
        vocab = []
        for idx in range(self.embeddings.shape[0]):
            tokens = self.tokenizer.convert_ids_to_tokens([idx])
            assert len(tokens) == 1
            token = tokens[0]
            if not token in self: 
                continue
            vocab.append(token)
        return vocab

    @staticmethod
    def get_embeddings(model):
        keys = list(model.state_dict().keys())
        funcs = [\
                lambda x: "embedding" in x and not "position" in x,
                lambda x: "word_embedding" in x,
                lambda x: "encoder.embed_tokens" in x]

        for func in funcs:
            filt_keys = list(filter(func, keys))
            if len(filt_keys) == 1:
                return model.state_dict()[filt_keys[0]].detach().numpy()

        raise Exception("Did not find unique embedding key: {}".format(\
                ",".join(keys)))
    
    @property
    def all_special_tokens(self):
        return self.tokenizer.all_special_tokens

    def index(self, word):
        if not word in self:
            raise KeyError(f"{word} not in embedding")
        return self.tokenizer.convert_tokens_to_ids([word])[0]

    def __init__(self, name):
        model_class, tokenizer_class = self.get_model_and_tokenizer_class(name)
        model = model_class.from_pretrained(name)

        self.tokenizer = tokenizer_class.from_pretrained(name)
        self.embeddings = self.get_embeddings(model)
        del model
        gc.collect()

    def __contains__(self, word):
        if word in self.all_special_tokens:
            return True
        
        # we return True if word is a special token
        # however, we return False if it BECOMES a special token during tokenization
        # (e.g., because it is too long and the tokenizer returns [UNK])
        
        if word.startswith("<extra_id_"):
            return False

        ids = self.tokenizer.convert_tokens_to_ids([word])
        assert len(ids) == 1
        return ids[0] not in self.tokenizer.all_special_ids
        
    def getvector(self, word):
        return self.getvectors([word])[0]

    def getvectors(self, words):
        return self.embeddings[[self.index(word) for word in words]]
