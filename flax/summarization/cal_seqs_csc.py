import logging
import os
import sys

import numpy as np
import pandas as pd

import jax
import jax.numpy as jnp
import optax
import transformers
from Bio import Seq, SeqIO
from dataclasses import dataclass, field
from typing import Callable, Optional
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    FlaxT5ForConditionalGeneration,
    HfArgumentParser,
    TrainingArguments,
)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--model-name-or-path', type=str, default=None,
                        help='The model checkpoint for weights initialization.')
    parser.add_argument('--config-name', type=str, default=None,
                        help='Pretrained config name or path if not the same as model_name.')
    parser.add_argument('--tokenizer-name', type=str, default=None,
                        help='Pretrained tokenizer name or path if not the same as model_name.')
    parser.add_argument('--base-fasta-filename', type=str, default=None,
                        help='the reference protein sequence to calculate csc')
    parser.add_argument('--cal-fasta-filename', type=str, default=None,
                        help='the protein finename you want to calculate its csc')
    parser.add_argument('--protein', type=str, default='s',
                        help='s1, s2 or s')
    parser.add_argument('--grammer', type=bool, default=False,
                        help='calculate the grammer change simultaneously')
    parser.add_argument('--merge', type=bool, default=False,
                        help='merge s1 and s2 csc results')
    parser.add_argument('--results-fname', type=str, default='',
                        help='results output file name')
    
    args = parser.parse_args()
    return args

def get_mutations(seq1, seq2):
    """
    This function is a copy and modification of <https://github.com/brianhie/viral-mutation/blob/81c80d41671670eb58cc46e957a1b0c4bf14856a/bin/cov.py#L273>`__ .
    """
    mutations = []
    from Bio import pairwise2
    alignment = pairwise2.align.globalms(
        seq1, seq2, 5, -4, -3, -.1, one_alignment_only=True,
    )[0]
    pos = 0
    for ch1, ch2 in zip(alignment[0], alignment[1]):
        if ch1 != ch2 and ch1 != '-' and ch2 != '-':
            mutations.append('{}{}{}'.format(ch1, pos + 1, ch2))
        if ch1 == '-' and ch2 != '-':
            mutations.append('{}ins{}'.format(pos + 1, ch2))
        if ch1 != '-' and ch2 == '-':
            mutations.append('{}{}del'.format(ch1, pos + 1))
        if ch1 != '-':
            pos += 1
    return list(set(mutations))


def get_embedding(model, config, tokenizer, shift_tokens_right_fn, vocabulary, seq):
    demo_output = ['X' for _ in range(len(seq))]
    inputs = tokenizer(seq, return_tensors="np")
    labels = tokenizer(''.join(demo_output), return_tensors="np")
    inputs["labels"] = labels["input_ids"]
    decoder_input_ids = shift_tokens_right_fn(
                jnp.array(labels["input_ids"]), config.pad_token_id, config.decoder_start_token_id
            )
    inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)
    # encoder_outputs = model.encode(**inputs)
    inputs["decoder_attention_mask"] = labels["attention_mask"]
    inputs.pop("labels")

    model_outputs = model(**inputs, params=model.params, train=False)
    log_probs = jax.nn.softmax(model_outputs[0])
    vocab_dict =log_probs[0][1:len(seq)+1,3:26]

    word_pos_prob = {}
    for pos in range(len(seq)):
        for i, word in enumerate(vocabulary):
            word_pos_prob[(pos,word.upper())] = float(vocab_dict[pos][i])

    decoder_hidden_states = model_outputs[1][-1]
    base_embedding = decoder_hidden_states.reshape(len(seq)+2, 512)

    return base_embedding, word_pos_prob


def cal_prob(protein, base_seq, decoded_preds, word_pos_prob):
    probs = []
    pred_mutations = []
    for index, mut_seq in enumerate(decoded_preds):
        mutations = get_mutations(base_seq, mut_seq)
        real_mutations = []
        if len(mutations) == 0:
            probs.append(0)
        else:
            mut_probs = []
            for mutation in mutations:
                if 'del' in mutation or 'ins' in mutation:
                    continue
                pos = int(mutation[1:-1]) - 1
                mut = mutation[-1]
                mut_probs.append(word_pos_prob[(pos, mut)])
            if protein == 's2':
                for mutation in mutations:
                    if 'ins' in mutation:
                        real_mutations.append(str(int(mutation[:-4])+688)+mutation[-4:])
                    elif 'del' in mutation:
                        real_mutations.append(mutation[0]+str(int(mutation[1:-3])+688)+'del')
                    else:
                        real_mutations.append(mutation[0]+str(int(mutation[1:-1])+688)+mutation[-1])
            else:
                real_mutations = mutations
            probs.append(np.mean(mut_probs))
        pred_mutations.append(' '.join(real_mutations))

    return probs, pred_mutations


def read_fasta(fasta_name):
    data = {'id': [], 'sequence': []}
    with open(fasta_name, 'r') as f:
        seq = ''
        key = ''
        for line in f:
            line = line.strip()
            if line[0] == '>':
                if key:
                    data['id'].append(key)
                    data['sequence'].append(seq)
                    seq = ''
                key = line[1:]
            else:
                seq += line
        if key:
            data['id'].append(key)
            data['sequence'].append(seq)

    data_df = pd.DataFrame(data)
    return data_df


def main():
    args = parse_args()
    logger = logging.getLogger(__name__)
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(f"args: {args}")
    base = str(SeqIO.read(args.base_fasta_filename, 'fasta').seq)
    if args.protein == 's1':
        base_seq = base[:688]
    elif args.protein == 's2':
        base_seq = base[688:]
    elif args.protein == 's':
        base_seq = base
    else:
        logging.error("not specify the protein you want to predict: s, s1 or s2")
        raise ValueError("please specify the protein you want to predict: s , s1 or s2")
    
    data_df = read_fasta(args.cal_fasta_filename)
    print('load data {} size: '.format(args.cal_fasta_filename), data_df.shape[0])

    logger.info(f" ------------------------load-model------------------------------")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
    config = AutoConfig.from_pretrained(args.config_name)
    config.output_hidden_states=True
    model = FlaxAutoModelForSeq2SeqLM.from_pretrained(args.model_name_or_path,\
        config = config)

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    """
    Important: do not change the order of the vocabulary.
    """
    vocabulary = ['▁','l','s','t','v','n','g','a','f',
            'i','q','d','k','p','y','e','r','c',
            'h','m','w','x','*']

    base_embedding, word_pos_prob = get_embedding(model, config, tokenizer, shift_tokens_right_fn, \
        vocabulary, base)


    s_list = data_df['sequence'].to_list()
    id_list = data_df['id'].to_list()
    semantic_list = []
    semantic_v2_list = []

    for index in range(len(id_list)):
        demo_output = ['X' for _ in range(len(base_seq))]
        inputs = tokenizer(s_list[index], return_tensors="np")
        labels = tokenizer(''.join(demo_output), return_tensors="np")
        inputs["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right_fn(
                    jnp.array(labels["input_ids"]), config.pad_token_id, config.decoder_start_token_id
                )
        inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)
        inputs["decoder_attention_mask"] = labels["attention_mask"]
        inputs.pop("labels")

        model_outputs = model(**inputs, params=model.params, train=False)

        decoder_hidden_states = model_outputs[1][-1]
        mut_embedding = decoder_hidden_states.reshape(len(base_seq)+2, 512)

        mut_change = np.sum(np.abs(mut_embedding.mean(0) - base_embedding.mean(0)))
        mut_change_v2 = np.linalg.norm(mut_embedding - base_embedding)

        semantic_list.append(mut_change)
        semantic_v2_list.append(mut_change_v2)
        print('id: {}'.format(id_list[index]), 'semantic: ',mut_change, 'semantic_v2: ',mut_change_v2)

    if args.grammer:
        grammer_list, _ = cal_prob(args.protein, base, s_list, word_pos_prob)
        print('finish calculate grammer')
        data = {'id':id_list, 'sequence':s_list, 'semantic':semantic_list, \
            'semantic_v2': semantic_v2_list, 'grammer':grammer_list}
    else:
        data = {'id':id_list, 'sequence':s_list, 'semantic':semantic_list, \
            'semantic_v2': semantic_v2_list}
    data_df = pd.DataFrame(data)
    data_df.to_csv(args.results_fname, index=0)
    print('all results are saved to {}'.format(args.results_fname))

if __name__ == "__main__":
    main()
