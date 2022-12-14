#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team All rights reserved.
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
"""
Calculate CSC Grammer && Predict mutations
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

from lib2to3.pgen2 import grammar
import logging
import os
import sys
import time
import traceback
import re
from Bio import BiopythonWarning
from Bio import Seq, SeqIO
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Callable, Optional

import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from pandas import Series,DataFrame
import pandas as pd
from datasets import Dataset, load_dataset, load_metric
from tqdm import tqdm

import jax
import jax.numpy as jnp
import optax
import sys
import transformers
from filelock import FileLock
from flax import jax_utils, traverse_util
from flax.jax_utils import pad_shard_unpad, unreplicate
from flax.training import train_state
from flax.training.common_utils import get_metrics, onehot, shard, shard_prng_key
from huggingface_hub import Repository
from transformers import (
    CONFIG_MAPPING,
    FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoTokenizer,
    FlaxAutoModelForSeq2SeqLM,
    HfArgumentParser,
    TrainingArguments,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
logger = logging.getLogger(__name__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)


MODEL_CONFIG_CLASSES = list(FLAX_MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization."
            "Don't set if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    dtype: Optional[str] = field(
        default="float32",
        metadata={
            "help": "Floating-point format in which the model weights should be initialized and trained. Choose one of `[float32, float16, bfloat16]`."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input predict data file to do prediction on (a text file)."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the `max_length` param of `model.generate`, which is used "
            "during evaluation."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate"}
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to `model.generate`, "
            "which is used during evaluation."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    csc: bool = field(
        default=True, metadata={"help": "Calculate csc score"}
    )

    grammer: bool = field(
        default=True, metadata={"help": "Calculate grammer"}
    )

    mutation: bool = field(
        default=True, metadata={"help": "get mutation between input and predicted sequence"}
    )

    baseline_fname: Optional[str] = field(
        default=None, metadata={"help": "Infrence protein sequence filename"}
    )

    generate_file_name: Optional[str] = field(
        default=None, metadata={
            "help": "Predicted protein sequence, csc and grammer output file name."
        },
    )

    protein: Optional[str] = field(
        default=None, metadata={"help": "Infrence s, s1 or s2"}
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length





class TrainState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


class PredictState(train_state.TrainState):
    dropout_rng: jnp.ndarray

    def replicate(self):
        return jax_utils.replicate(self).replace(dropout_rng=shard_prng_key(self.dropout_rng))


def data_loader(rng: jax.random.PRNGKey, dataset: Dataset, batch_size: int, shuffle: bool = False):
    """
    Returns batches of size `batch_size` from truncated `dataset`, sharded over all local devices.
    Shuffle batches if `shuffle` is `True`.
    """
    steps_per_epoch = len(dataset) // batch_size

    if shuffle:
        batch_idx = jax.random.permutation(rng, len(dataset))
    else:
        batch_idx = jnp.arange(len(dataset))

    batch_idx = batch_idx[: steps_per_epoch * batch_size]  # Skip incomplete batch.
    batch_idx = batch_idx.reshape((steps_per_epoch, batch_size))

    for idx in batch_idx:
        batch = dataset[idx]
        batch = {k: jnp.array(v) for k, v in batch.items()}

        batch = shard(batch)

        yield batch


def write_metric(summary_writer, train_metrics, eval_metrics, train_time, step):
    summary_writer.scalar("train_time", train_time, step)

    train_metrics = get_metrics(train_metrics)
    for key, vals in train_metrics.items():
        tag = f"train_{key}"
        for i, val in enumerate(vals):
            summary_writer.scalar(tag, val, step - len(vals) + i + 1)

    for metric_name, value in eval_metrics.items():
        summary_writer.scalar(f"eval_{metric_name}", value, step)


def create_learning_rate_fn(
    train_ds_size: int, train_batch_size: int, num_train_epochs: int, num_warmup_steps: int, learning_rate: float
) -> Callable[[int], jnp.array]:
    """Returns a linear warmup, linear_decay learning rate function."""
    steps_per_epoch = train_ds_size // train_batch_size
    num_train_steps = steps_per_epoch * num_train_epochs
    warmup_fn = optax.linear_schedule(init_value=0.0, end_value=learning_rate, transition_steps=num_warmup_steps)
    decay_fn = optax.linear_schedule(
        init_value=learning_rate, end_value=0, transition_steps=num_train_steps - num_warmup_steps
    )
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn], boundaries=[num_warmup_steps])
    return schedule_fn


def get_mutations(seq1, seq2):
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

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty."
            "Use --overwrite_output_dir to overcome."
        )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    # Setup logging, we only want one process per machine to log things on the screen.
    logger.setLevel(logging.INFO if jax.process_index() == 0 else logging.ERROR)
    if jax.process_index() == 0:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")


    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        dataset = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir, keep_in_memory=False
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        dataset = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer

    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, cache_dir=model_args.cache_dir)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, cache_dir=model_args.cache_dir)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    config.output_hidden_states=True

    if model_args.tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    elif model_args.model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path, cache_dir=model_args.cache_dir, use_fast=model_args.use_fast_tokenizer
        )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if model_args.model_name_or_path:
        model = FlaxAutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path, config=config , seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )
    else:
        model = FlaxAutoModelForSeq2SeqLM.from_config(
            config, seed=training_args.seed, dtype=getattr(jnp, model_args.dtype)
        )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    """
    Important: do not change the order of the vocabulary.
    """
    vocabulary = ['▁','l','s','t','v','n','g','a','f',
            'i','q','d','k','p','y','e','r','c',
            'h','m','w','x','*']

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = dataset["train"].column_names
    elif training_args.do_eval:
        column_names = dataset["validation"].column_names
    elif training_args.do_predict:
        column_names = dataset["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    # In Flax, for seq2seq models we need to pass `decoder_input_ids`
    # as the Flax models don't accept `labels`, we need to prepare the decoder_input_ids here
    # for that dynamically import the `shift_tokens_right` function from the model file
    model_module = __import__(model.__module__, fromlist=["shift_tokens_tight"])
    shift_tokens_right_fn = getattr(model_module, "shift_tokens_right")

    # Setting padding="max_length" as we need fixed length inputs for jitted functions
    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        inputs = [prefix + inp for inp in inputs]
        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding="max_length", truncation=True, return_tensors="np"
        )

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding="max_length", truncation=True, return_tensors="np"
            )

        model_inputs["labels"] = labels["input_ids"]
        decoder_input_ids = shift_tokens_right_fn(
            jnp.array(labels["input_ids"]), config.pad_token_id, config.decoder_start_token_id
        )
        model_inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)

        # We need decoder_attention_mask so we can ignore pad tokens from loss
        model_inputs["decoder_attention_mask"] = labels["attention_mask"]

        return model_inputs

    if training_args.do_train:
        if "train" not in dataset:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = dataset["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on train dataset",
        )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = dataset["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on validation dataset",
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = dataset["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        predict_dataset = predict_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on prediction dataset",
        )


    # Initialize our training
    rng = jax.random.PRNGKey(training_args.seed)
    rng, dropout_rng = jax.random.split(rng)

    # Store some constant
    num_epochs = int(training_args.num_train_epochs)
    train_batch_size = int(training_args.per_device_train_batch_size) * jax.device_count()
    eval_batch_size = int(training_args.per_device_eval_batch_size) * jax.device_count()
    # steps_per_epoch = len(train_dataset) // train_batch_size
    # total_train_steps = steps_per_epoch * num_epochs

    # Create learning rate schedule
    linear_decay_lr_schedule_fn = create_learning_rate_fn(
        128,
        train_batch_size,
        training_args.num_train_epochs,
        training_args.warmup_steps,
        training_args.learning_rate,
    )

    # We use Optax's "masking" functionality to not apply weight decay
    # to bias and LayerNorm scale parameters. decay_mask_fn returns a
    # mask boolean with the same structure as the parameters.
    # The mask is True for parameters that should be decayed.
    # Note that this mask is specifically adapted for FlaxBart.
    # For FlaxT5, one should correct the layer norm parameter naming
    # accordingly - see `run_t5_mlm_flax.py` e.g.
    def decay_mask_fn(params):
        flat_params = traverse_util.flatten_dict(params)
        layer_norm_params = [
            (name, "scale") for name in ["self_attn_layer_norm", "layernorm_embedding", "final_layer_norm"]
        ]
        flat_mask = {path: (path[-1] != "bias" and path[-2:] not in layer_norm_params) for path in flat_params}
        return traverse_util.unflatten_dict(flat_mask)

    # create adam optimizer
    adamw = optax.adamw(
        learning_rate=linear_decay_lr_schedule_fn,
        b1=training_args.adam_beta1,
        b2=training_args.adam_beta2,
        eps=training_args.adam_epsilon,
        weight_decay=training_args.weight_decay,
        mask=decay_mask_fn,
    )

    # Setup Predict state
    state = PredictState.create(apply_fn=model.__call__, params=model.params, tx=adamw, dropout_rng=dropout_rng)

    def cal_csc(base_embedding ,mut_embedding):
        mut_change = np.sum(np.abs(mut_embedding.mean(0) - base_embedding.mean(0)))
        return mut_change

    # Define eval fn
    def eval_step(params, batch):
        labels = batch.pop("labels")
        outputs = model(**batch, params=params, train=False)
        log_probs = jax.nn.softmax(outputs[0])
        decoder_hidden_states = outputs[1][-1]
        return log_probs, decoder_hidden_states

    # Replicate the train state on each device
    state = state.replicate()

    
    # Define generation function
    max_length = (
        data_args.val_max_target_length if data_args.val_max_target_length is not None else model.config.max_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else model.config.num_beams
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
    
    # calculate prob
    base_seq = str(SeqIO.read(data_args.baseline_fname, 'fasta').seq)
    if data_args.protein == 's1':
        base_seq = base_seq[:688]
    elif data_args.protein == 's2':
        base_seq = base_seq[688:]
    elif data_args.protein == 's':
        base_seq = base_seq
    else:
        logging.error("not specify the protein you want to predict: s, s1 or s2")
        raise ValueError("please specify the protein you want to predict: s , s1 or s2")
    print('---------length base_seq--------',len(base_seq))
    demo_output = ['X' for _ in range(max_length)]
    inputs = tokenizer(base_seq, return_tensors="np")
    labels = tokenizer(''.join(demo_output), return_tensors="np")
    inputs["labels"] = labels["input_ids"]
    decoder_input_ids = shift_tokens_right_fn(
                jnp.array(labels["input_ids"]), config.pad_token_id, config.decoder_start_token_id
            )
    inputs["decoder_input_ids"] = np.asarray(decoder_input_ids)
    inputs["decoder_attention_mask"] = labels["attention_mask"]
    inputs.pop("labels")

    model_outputs = model(**inputs, params=model.params, train=False)
    log_probs = jax.nn.softmax(model_outputs[0])
    vocab_dict =log_probs[0][1:len(base_seq)+1,3:26]

    word_pos_prob = {}
    for pos in range(len(base_seq)):
        for i, word in enumerate(vocabulary):
            word_pos_prob[(pos,word.upper())] = float(vocab_dict[pos][i])

    base_embedding = model_outputs[1][-1][:,1:-1,]

    # print('base_embedding.shape------',base_embedding.shape)


    def cal_prob(protein, base_seq, decoded_preds, word_pos_prob):
        probs = []
        pred_mutations = []
        for index, pred_sequence in enumerate(decoded_preds):
            mut_seq = pred_sequence
            # assert len(base_seq) == len(pred_sequence)
            mutations = get_mutations(base_seq, mut_seq)
            real_mutations = []
            # print('mutations: ',mutations)
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
                # probs.append(np.mean(np.log10(mut_probs)))
                probs.append(np.mean(mut_probs))
            pred_mutations.append(' '.join(real_mutations))

        return probs, pred_mutations

    def find_con(s):
        l1 = []
        res = []
        for x in sorted(set(s)):
            l1.append(x)
            if x+1 not in s:
                if len(l1) != 1 and len(l1) <=5:
                    res.append(l1)
                l1 = []
        return res

    def post_process_mutations(mutations):
        new_mutations = []
        for m in mutations:
            real_ms = []
            dels = []
            real_dels = []
            try:
                temp = m.split(' ')
                for t in temp:
                    if 'del' not in t and 'X' not in t:
                        real_ms.append(t)
                    elif 'del' in t and 'X' not in t:
                        dels.append(t)
                sk = []
                for k in dels:
                    sk.append(int(k[1:-3]))
                bid = find_con(sk)
                con_sk = []
                for bi in bid:
                    con_sk.extend(bi)
                for d in dels:
                    if int(d[1:-3]) in con_sk:
                        real_dels.append(d)
                real_ms.extend(real_dels)
                new_mutations.extend(real_ms)
            except:
                continue
        return ' '.join(list(set(new_mutations)))

    def generate_step(params, batch):
        model.params = params
        output_ids = model.generate(batch["input_ids"], attention_mask=batch["attention_mask"], **gen_kwargs)
        return output_ids.sequences
    # p_eval_step = jax.pmap(partial(eval_step, label_smoothing_factor=training_args.label_smoothing_factor), "batch")
    p_eval_step = jax.pmap(eval_step, "batch")
    p_generate_step = jax.pmap(generate_step, "batch")
        
    # Create sampling rng
    rng, input_rng = jax.random.split(rng)
    # params = jax.device_get(jax.tree_map(lambda x: x[0], model.params))

    if training_args.do_predict:
        logger.info("*** Predict ***")

        # pred_metrics = []
        pred_generations = []
        csc = []

        pred_loader = data_loader(input_rng, predict_dataset, eval_batch_size)
        pred_steps = len(predict_dataset) // eval_batch_size
        for _ in tqdm(range(pred_steps), desc="Predicting...", position=2, leave=False):
            # Model forward
            batch = next(pred_loader)

            if data_args.predict_with_generate:
                # generated_ids = pad_shard_unpad(p_generate_step)(model.params, batch)
                generated_ids = p_generate_step(state.params, batch)
                log_probs, decoder_hidden_states = p_eval_step(state.params, batch)
                mut_embedding = np.squeeze(decoder_hidden_states)
                batch_csc2 = np.sqrt(np.sum((np.abs(base_embedding.mean(1) - mut_embedding.mean(1)))**2+1e-8, axis=-1))
                csc.extend(batch_csc2)
                pred_generations.extend(jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])))
                batch_shape = jax.device_get(generated_ids.reshape(-1, gen_kwargs["max_length"])).shape
                logger.info(f" ------------------------------------------------------batch size: {batch_shape[0]}")


        if data_args.predict_with_generate:
            #=== Decode ===
            decoded_preds = []
            print('------------generation count:',len(pred_generations))
            decoded_preds_low = tokenizer.batch_decode(pred_generations, skip_special_tokens=True)
            for pred in decoded_preds_low:
                if len(base_seq) > len(pred):
                    pred = pred + ''.join(['X' for _ in range(len(base_seq)-len(pred))])
                elif len(base_seq) < len(pred):
                    pred = pred[:-(len(pred) - len(base_seq))]
                decoded_preds.append(pred.upper())
        

        # === Grammer ===
        logger.info(f" ------------------------------------------------------calculate grammer.")
        grammaticality, pred_mutations = cal_prob(data_args.protein, base_seq, decoded_preds, word_pos_prob)
        pred_mutations = post_process_mutations(pred_mutations)

        # print('=========',len(decoded_preds), len(pred_mutations),len(csc),len(grammaticality))


        if data_args.grammer and data_args.csc:
            results = {'pred_sequences':decoded_preds,'pred_mutations':pred_mutations,'csc':csc,'grammer':grammaticality}
        else:
            results = {'pred_sequences': decoded_preds,'pred_mutations': pred_mutations}

        
        results_df = DataFrame(results)
        results_file_name = data_args.generate_file_name
        results_df.to_csv(results_file_name)

        logger.info(f" All prediction sequences, CSC socre and grammer all saved to {results_file_name}")

if __name__ == "__main__":
    main()
