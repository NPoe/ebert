# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Adapted from https://github.com/thunlp/ERNIE/blob/master/code/run_fewrel.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
from tqdm import tqdm, trange
import json

import string
import re

from embeddings import *
from mappers import *

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import copy

from utils.util_wikidata import wikidata2title
from pytorch_transformers import BertTokenizer, AdamW, WarmupLinearSchedule, BertForSequenceClassification
import sys

from pytorch_transformers import PreTrainedModel
from emb_input_transformers import EmbInputBertForSequenceClassification, EmbInputBertModel
from torch import nn

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
    
    @classmethod
    def _read_json(cls, input_file):
        with open(input_file, "r", encoding='utf-8') as f:
            return json.loads(f.read())

class FewrelProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line["ents"]:
                if x[1] == 1:
                    x[1] = 0

            text_a = (line['text'], line["ents"], line["ents"])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples



def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, threshold, patterns, ent_type, embedding):
    """Loads a data file into a list of `InputBatch`s."""
   
    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    all_wikidata_ids = set()
    for (ex_index, example) in enumerate(examples):
        all_wikidata_ids.update([x[0] for x in example.text_a[2]])

    mapping = wikidata2title(all_wikidata_ids, verbose = 1)

    for (ex_index, example) in enumerate(examples):
        ex_text_a = example.text_a[0]
        ents = example.text_a[1]
        ann = example.text_a[2]
        
        for x in ann:
            title_escaped = None
            for title in sorted(mapping[x[0]]):
                if "ENTITY/" + title in embedding:
                    title_escaped = "q" + title + "q"
                    for i, p in enumerate(string.punctuation):
                        title_escaped = title_escaped.replace(p, "q" + str(i) + "q") 
                        # we replace all punctuation in the title, so that the basic tokenizer does not destroy it
                    if not title_escaped in tokenizer.vocab:
                        tokenizer.vocab[title_escaped] = len(tokenizer.vocab)
                        tokenizer.ids_to_tokens[tokenizer.vocab[title_escaped]] = "ENTITY/" + title
                    
                    assert tokenizer.ids_to_tokens[tokenizer.vocab[title_escaped]] == "ENTITY/" + title
                    break

            x[0] = title_escaped

            
        ann = [x for x in ann if x[-1] > threshold and x[0] is not None]  
        ann.sort(key = lambda x:x[1], reverse = True)
        
        if isinstance(patterns, str):
            patterns = [patterns] * len(ents)
        assert len(patterns) == len(ents)
        
        ents_and_patterns = list(zip(ents, patterns))
        ents_and_patterns.sort(key = lambda x:x[0][1], reverse = True)

        offsets = {}

        for x in ann:
            if ent_type == "concat":
                name = x[0] + " / " + ex_text_a[x[1]:x[2]]
            elif ent_type == "replace":
                name = x[0]
            elif ent_type == "none":
                name = ex_text_a[x[1]:x[2]]
            else:
                raise Exception("Unknown ent type")

            ex_text_a = ex_text_a[:x[1]] + name + ex_text_a[x[2]:]
            offsets[x[1]] = len(name) - (x[2]-x[1])
        
        for ent, pattern in ents_and_patterns:
            for off in offsets:
                if off < ent[1]:
                    ent[1] += offsets[off]
                if off <= ent[2]:
                    ent[2] += offsets[off]

            name = ex_text_a[ent[1]:ent[2]]
            assert "{name}" in pattern
            name = pattern.replace("{name}", name)
            ex_text_a = ex_text_a[:ent[1]] + name + ex_text_a[ent[2]:]
        
        tokens_a = tokenizer.tokenize(ex_text_a)
        assert example.text_b is None
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        padding_ = [-1] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        
        if isinstance(example.label, list):
            label_id = [0]*len(label_map)
            for l in example.label:
                label_id[label_map[l]] = 1
        else:
            label_id = label_map[example.label]

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            if isinstance(example.label, list):
                logger.info("label: %s (id = %s)" % (" ".join(example.label), " ".join([str(x) for x in label_id])))
            else:
                logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features

class TacredProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        labels = set([x.label for x in examples])
        return examples, list(labels)

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")

    def get_labels(self):
        """Useless"""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            for x in line['ents']:
                if x[1] == 1:
                    x[1] = 0
                    #print(line['text'][x[1]:x[2]].encode("utf-8"))

            text_a = (line['text'], line['ents'], line['ann'])
            label = line['label']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        
            
        return examples



def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

class TypingProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.json")))
        examples = self._create_examples(
            self._read_json(os.path.join(data_dir, "train.json")), "train")
        d = {}
        for e in examples:
            for l in e.label:
                if l in d:
                    d[l] += 1
                else:
                    d[l] = 1
        for k, v in d.items():
            d[k] = (len(examples) - v) * 1. /v

        return examples, list(d.keys())

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "dev.json")), "dev")
    
    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_json(os.path.join(data_dir, "test.json")), "test")


    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = i
            text_a = (line['sent'], [["SPAN", line["start"], line["end"]]], line['ents'])
            label = line['labels']
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0


class EmbInputBertForEntityTyping(BertForSequenceClassification):
    def __init__(self, config):
        super(EmbInputBertForEntityTyping, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = EmbInputBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.typing = nn.Linear(config.hidden_size, self.num_labels, False)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, labels=None):
        pooled_output = self.bert(input_ids=input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids, 
                position_ids=position_ids,
                head_mask=head_mask)[1]
        
        pooled_output = self.dropout(pooled_output)
        logits = self.typing(pooled_output)
        
        
        if labels is not None:
            labels = labels.to(dtype = logits.dtype)
            
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1, self.num_labels))
            return loss, logits
        else:
            return logits

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--ernie_model", default="bert-base-cased", type=str,
                        help="Ernie pre-trained model")
    parser.add_argument("--embedding", default="wikipedia2vec-base-cased", type=str,
                        help="Embeddings")
    parser.add_argument("--mapper", default="wikipedia2vec-base-cased.bert-base-cased.linear", type=str,
                        help="Embeddings")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=256,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--ent", 
                        required=True,
                        choices = ("none", "concat", "replace"),
                        help="How to use entity embeddings.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=4,
                        type=int,
                        help="Total batch size for evaluation.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=10.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        default=False,
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--threshold', type=float, default=.3)

    args = parser.parse_args()

    processors = FewrelProcessor
    num_labels_task = 80

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval and not args.do_test:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)


    processor = processors()
    num_labels = num_labels_task
    label_list = None

    embedding = MappedEmbedding(load_embedding(args.embedding), load_mapper(args.mapper))
    model_embedding = load_embedding(args.ernie_model)
    tokenizer = BertTokenizer.from_pretrained(args.ernie_model)
    
    train_examples = None
    num_train_steps = None
    train_examples, label_list = processor.get_train_examples(args.data_dir)
    
    label_list = sorted(label_list, key = lambda x:int(x[1:]))

    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    model = EmbInputBertForSequenceClassification.from_pretrained(args.ernie_model, num_labels = num_labels, output_attentions = True)

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_proportion*t_total, t_total=t_total)
    
    global_step = 0
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    
    patterns = ["# {name} #", "$ {name} $"]
    
    if args.do_train:

        train_features = convert_examples_to_features(
            train_examples, label_list, args.max_seq_length, tokenizer, args.threshold, patterns=patterns, ent_type=args.ent,
            embedding = embedding)
        
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_examples))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        output_loss_file = os.path.join(args.output_dir, "loss")
        loss_fout = open(output_loss_file, 'w')
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                input_words = tokenizer.convert_ids_to_tokens(batch[0].cpu().numpy().flatten())
                input_vecs = [embedding[w] if w.startswith("ENTITY/") else model_embedding[w] for w in input_words]
                input_vecs = np.array(input_vecs).reshape(batch[0].shape + (-1,))
                
                batch[0] = torch.tensor(input_vecs)
                
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, label_ids = batch

                loss = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids, labels=label_ids)[0]
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                loss_fout.write("{}\n".format(loss.item()))
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
            model_to_save = model.module if hasattr(model, 'module') else model
            output_model_file_step = os.path.join(args.output_dir, "pytorch_model.bin_{}".format(global_step))
            torch.save(model_to_save.state_dict(), output_model_file_step)

        # Save a trained model
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)
   
    if args.do_eval or args.do_test:
        del model
        output_model_files = [f for f in os.listdir(args.output_dir) if f.startswith("pytorch_model.bin")]
        #output_model_files = ["pytorch_model.bin"] #TODO

        for output_model_file in output_model_files:

            model = EmbInputBertForSequenceClassification.from_pretrained(args.ernie_model, num_labels = num_labels)
        
            model.load_state_dict(torch.load(os.path.join(args.output_dir, output_model_file)))
            if args.fp16:
                model.half()
            model.to(device)
            model.eval()

            dsets = []
            if args.do_eval:
                dsets.append((processor.get_dev_examples, "eval"))
            if args.do_test:
                dsets.append((processor.get_test_examples, "test"))

            for dset_func, dset_name in dsets:
                features = convert_examples_to_features(
                    dset_func(args.data_dir), label_list, args.max_seq_length, tokenizer, 
                    args.threshold, patterns=patterns, ent_type=args.ent, embedding = embedding)

                step = output_model_file.replace("pytorch_model.bin", "")

                fpred = open(os.path.join(args.output_dir, dset_name + f"_pred{step}.txt"), "w")
                fgold = open(os.path.join(args.output_dir, dset_name + f"_gold{step}.txt"), "w")
                
                fwords = open(os.path.join(args.output_dir, dset_name + f"_words{step}.txt"), "w")

                all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
                all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
                all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
                data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        
                dataloader = DataLoader(data, sampler=None, batch_size=args.eval_batch_size)
           
                acc = []
                all_probs = []

                for step, batch in enumerate(tqdm(dataloader, desc="Evaluation {} {}".format(output_model_file, dset_name))):
                    input_words = tokenizer.convert_ids_to_tokens(batch[0].cpu().numpy().flatten())
                    input_vecs = [embedding[w] if w.startswith("ENTITY/") else model_embedding[w] for w in input_words]
                    input_vecs = np.array(input_vecs).reshape(batch[0].shape + (-1,))

                    batch[0] = torch.tensor(input_vecs)
                
                    batch = tuple(t.to(device) for t in batch)
                    input_ids, input_mask, segment_ids, label_ids = batch
                    
                    logits = model(input_ids, attention_mask=input_mask, token_type_ids=segment_ids)[0]
                    prob = torch.softmax(logits, -1)
                    all_probs.append(prob.detach().cpu().numpy())

                    predictions = prob.argmax(-1)
                    for a, b in zip(predictions, label_ids):
                        fgold.write("{}\n".format(label_list[b]))
                        fpred.write("{}\n".format(label_list[a]))

if __name__ == "__main__":
    main()
