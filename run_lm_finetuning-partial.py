# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.


NOTES:
    - Add the parse as a prefix. Need to make sure tokenization works exactly as desired, and only measure perplexity for non-parse tokens.
    - Q: How does state work?
    - Q: Can we add a new feature where we first get the vector for the parse, then run an LM using the vector + vector of the text.
    - Q: Can we use transformer-xl?
"""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import glob
import logging
import os
import pickle
import random
import re
import shutil
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                                  BertConfig, BertForMaskedLM, BertTokenizer,
                                  GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                                  OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                  RobertaConfig, RobertaForMaskedLM, RobertaTokenizer,
                                  DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer,
                                  CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)


logger = logging.getLogger(__name__)


MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    'camembert': (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer)
}

my_globals = {}


def read_actions_vocab(path):
    vocab = {}
    vocab['PAD'] = len(vocab)
    vocab['BOS'] = len(vocab)
    vocab['EOS'] = len(vocab)
    with open(path) as f:
        for i, line in enumerate(f):
            action = line.strip()
            vocab[action] = len(vocab)
    return vocab

def tokenize_for_word_ids(tokenizer, sentence):
    tokens = tokenizer.tokenize(sentence)
    words = sentence.split(' ')
    i_word = 0
    tok_so_far = 0
    word_so_far = len(words[i_word])
    tok_to_word = []
    for tok in tokens:
        btok = tok.encode()
        if b'\xc4\xa0' in btok:
            if btok.startswith(b'\xc4\xa0'):
                btok = btok[2:]
            elif btok.endswithc(b'\xc4\xa0'):
                btok = btok[:-2]
            else:
                raise ValueError('Split should be at beg or end only: {}'.format(btok))
            tok = btok.decode()
        tok_so_far += len(tok)

        if tok_so_far > word_so_far:
            i_word += 1
            word_so_far += len(words[i_word])

        tok_to_word.append(i_word)
    return tok_to_word

def tokenize_for_action_ids(sentence, actions):
    actions = actions.split()
    words = sentence.split()
    word_to_action = []
    i_word = -1
    shifts = [a for a in actions if a == 'SHIFT']
    assert len(shifts) == len(words)
    for i_action, a in enumerate(actions):
        if a == 'SHIFT':
            if i_word >= 0:
                word_to_action.append(i_action - 1)
            i_word += 1
        if i_action == len(actions) - 1:
            word_to_action.append(i_action)
    assert len(word_to_action) == len(words), (len(word_to_action), len(words))
    return word_to_action

class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path='train', block_size=512):
        assert os.path.isfile(file_path)
        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(directory, args.model_name_or_path + '_cached_lm_' + str(block_size) + '_' + filename)
        action2idx = read_actions_vocab(args.actions_file)

        PADDING_TOKEN = '_'
        PADDING_ID = tokenizer.get_vocab()['_'] # TODO: Should we use a special token?
        SPLIT_TOKEN = '#'
        SPLIT_ID = tokenizer.get_vocab()['#'] # TODO: Should this be different from pad token?

        TI_ACTION = 0
        TI_FIRST_TOK = 1
        TI_TOK = 2
        TI_PAD = 3

        if False and os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, 'rb') as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)


            self.data = collections.defaultdict(list)
            self.stats = collections.Counter()
            skipped = 0

            def for_partial(sentence, actions, lookahead=True):
                tokens = sentence.split()
                actions = actions.split()
                assert len(tokens) == collections.Counter(actions)['SHIFT']

                new_seq, new_mask = [], []
                raw_seq = []
                block_tokens = []
                block_actions, tmp_a = [], []

                # Get blocks (action).
                i_tok = -1
                for i_action, a in enumerate(actions):
                    if a == 'SHIFT':
                        i_tok += 1
                        if i_tok > 0:
                            if len(tmp_a) > 0:
                                block_actions.append(tmp_a)
                            tmp_a = []

                    raw_tok = tokenizer.tokenize(a)
                    tmp_a += raw_tok

                if len(tmp_a) > 0:
                    block_actions.append(tmp_a)

                # Get blocks (token).
                for i_tok, tok in enumerate(sentence.split()):
                    if i_tok > 0:
                        tok = ' ' + tok
                    raw_tok = tokenizer.tokenize(tok)
                    block_tokens.append(raw_tok)

                assert len(block_actions) == len(block_tokens)
                assert len(block_tokens) == len(tokens)

                # Convert into output.
                if lookahead:
                    for i, (a, tok) in enumerate(zip(block_actions, block_tokens)):
                        new_seq += tokenizer.convert_tokens_to_ids(a)
                        new_mask += [TI_ACTION] * len(a)

                        new_seq += tokenizer.convert_tokens_to_ids(tok)
                        if i == 0:
                            new_mask += [TI_FIRST_TOK] * len(tok)
                        else:
                            new_mask += [TI_TOK] * len(tok)

                        raw_seq += a + tok
                else:
                    for i, (a, tok) in enumerate(zip(block_actions, block_tokens)):
                        # TOKEN
                        new_seq += tokenizer.convert_tokens_to_ids(tok)
                        if i == 0:
                            new_mask += [TI_FIRST_TOK] * len(tok)
                        else:
                            new_mask += [TI_TOK] * len(tok)

                        # ACTION
                        new_seq += tokenizer.convert_tokens_to_ids(a)
                        new_mask += [TI_ACTION] * len(a)

                        raw_seq += tok + a

                return new_seq, new_mask, raw_seq

            WRITE = file_path.endswith('dev-template.txt')

            if WRITE:
                f_data_lst = []

            with open(file_path, encoding="utf-8") as f:
                for line in tqdm(f):
                    prefix, sentence, tree, actions = line.split('_')
                    prefix = prefix.strip()
                    sentence = sentence.strip()
                    tree = tree.strip()
                    actions = actions.strip()
                    my_length = len(sentence.split())

                    # This is the pad token. Double check it is not predicted anywhere.
                    assert PADDING_TOKEN not in prefix and PADDING_TOKEN not in sentence, line
                    assert SPLIT_TOKEN not in prefix and SPLIT_TOKEN not in sentence, line

                    tokenized_text, mask, raw_seq = for_partial(sentence, actions, lookahead=not args.no_lookahead)

                    #if len(tokenized_text) > block_size:
                    #    skipped += 1
                    #    continue

                    if my_length > my_globals['max_length']:
                        skipped += 1
                        continue

                    assert len(tokenized_text) <= block_size

                    assert len(tokenized_text) == len(mask)

                    if len(tokenized_text) < block_size:
                        # pad left
                        # tokenized_text = [PADDING_ID] * (block_size - len(tokenized_text)) + tokenized_text
                        # pad right
                        tokenized_text = tokenized_text + [PADDING_ID] * (block_size - len(tokenized_text))
                        mask = mask + [TI_PAD] * (block_size - len(mask))

                    assert len(tokenized_text) == len(mask)

                    self.data['examples'].append(tokenizer.build_inputs_with_special_tokens(tokenized_text))
                    self.data['raw_seq'].append(raw_seq)
                    self.data['mask'].append(mask)
                    self.stats.update(mask)

                    if WRITE:
                        f_data_lst.append(line.strip())

            if WRITE:
                with open('dev-template-partial.txt', 'w') as f:
                    for line in f_data_lst:
                        f.write(line)
                        f.write('\n')

            logger.info("    SKIPPED = {}".format(skipped))
            logger.info("    STATS = {}".format(self.stats))

            #logger.info("Saving features into cached file %s", cached_features_file)
            #with open(cached_features_file, 'wb') as handle:
            #    pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.data['examples'])

    def __getitem__(self, item):
        return torch.tensor(self.data['examples'][item]), torch.tensor(self.data['mask'][item])


def load_and_cache_examples(args, tokenizer, evaluate=False):
    dataset = TextDataset(tokenizer, args, file_path=args.eval_data_file if evaluate else args.train_data_file, block_size=args.block_size)
    return dataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _rotate_checkpoints(args, checkpoint_prefix, use_mtime=False):
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    glob_checkpoints = glob.glob(os.path.join(args.output_dir, '{}-*'.format(checkpoint_prefix)))
    if len(glob_checkpoints) <= args.save_total_limit:
        return

    ordering_and_checkpoint_path = []
    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match('.*{}-([0-9]+)'.format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs, tokenizer, args):
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -1  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def train(args, train_dataset, model, tokenizer):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    if args.tree_mode == 'partial':
        action2idx = read_actions_vocab(args.actions_file)
        model.tree_encoder = TreeEncoder(action2idx)

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(os.path.join(args.model_name_or_path, 'scheduler.pt')):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model path
        global_step = int(args.model_name_or_path.split('-')[-1].split('/')[0])
        epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
        steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

        logger.info("  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("  Continuing training from epoch %d", epochs_trained)
        logger.info("  Continuing training from global step %d", global_step)
        logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

    tr_loss, logging_loss = 0.0, 0.0

    model_to_resize = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
    model_to_resize.resize_token_embeddings(len(tokenizer))

    PADDING_TOKEN = '_'
    PADDING_ID = tokenizer.get_vocab()['_'] # TODO: Should we use a special token?
    SPLIT_TOKEN = '#'
    SPLIT_ID = tokenizer.get_vocab()['#'] # TODO: Should this be different from pad token?

    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, (batch, token_mask) in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)
            model.train()
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
            # loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

            # START: Compute loss.
            #shift_logits = outputs[1][..., :-1, :].contiguous()
            #shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
            #vocab_size = shift_logits.shape[-1]
            #shift_logits = shift_logits.view(-1, vocab_size)
            #shift_labels = shift_labels.view(-1)
            #mask = shift_labels != PADDING_ID
            #loss = loss_fct(shift_logits[mask], shift_labels[mask])

            shift_logits = outputs[1][..., :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_mask = token_mask[:, 1:].contiguous() == 2
            loss = loss_fct(shift_logits[shift_mask], shift_labels[shift_mask])
            # END

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    checkpoint_prefix = 'checkpoint'
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, '{}-{}'.format(checkpoint_prefix, global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    _rotate_checkpoints(args, checkpoint_prefix)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()

    PADDING_TOKEN = '_'
    PADDING_ID = tokenizer.get_vocab()['_'] # TODO: Should we use a special token?
    SPLIT_TOKEN = '#'
    SPLIT_ID = tokenizer.get_vocab()['#'] # TODO: Should this be different from pad token?

    cache = collections.defaultdict(list)

    for batch, batch_mask in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)

            if True:
                # START: Compute loss.
                # TODO: This also masks the split token. Do we need to predict the split?
                #shift_logits = outputs[1][..., :-1, :].contiguous()
                #shift_labels = labels[..., 1:].contiguous()
                shift_logits = outputs[1][..., :-1, :].contiguous()
                shift_logits_argsort = shift_logits.argsort(dim=2, descending=True)
                shift_prob = torch.softmax(shift_logits, dim=2)
                shift_labels = labels[:, 1:].contiguous()
                shift_mask = batch_mask[:, 1:].contiguous() == 2
                ## Compute loss.
                loss_fct = nn.CrossEntropyLoss(ignore_index=-1)
                shift_labels = shift_labels[shift_mask]
                loss = loss_fct(shift_logits[shift_mask], shift_labels)
                lm_loss = loss
                # END

                assert shift_labels.shape[0] == shift_mask.long().sum().item()

                cache['num_bytes'].append(shift_mask.long().sum().item())
                cache['loss'].append(lm_loss.item())
                cache['size'].append(shift_labels.shape[0])

                #token_ids = tokenizer.convert_ids_to_tokens(shift_labels.view(-1).tolist())
                #s = tokenizer.convert_tokens_to_string(token_ids)
                #print(s)
                #print('')

                EXTRA = any([args.show_predictions])

                if EXTRA:

                    _labels = labels[:, 1:].contiguous()
                    _mask = batch_mask[:, 1:].contiguous() == 2
                    _first_mask = batch_mask[:, 1:].contiguous() == 1

                    for i_row, (row, mask_row, mask_row_0) in enumerate(zip(_labels.tolist(), _mask.tolist(), _first_mask.tolist())):
                        row_index = [i for i, (y, z) in enumerate(zip(mask_row, mask_row_0)) if y or z]
                        row_ids = tokenizer.convert_ids_to_tokens([row[i] for i in row_index])
                        _p = shift_prob[i_row].tolist()
                        row_probs = [_p[i] for i in row_index]
                        row_rank = shift_logits_argsort[i_row][row_index]
                        sent = tokenizer.convert_tokens_to_string(row_ids)

                        if args.show_predictions:
                            print(sent)
                            num_predictions = 10
                            for i_start in range(len(row_ids) - 1):
                                chunk = tokenizer.convert_tokens_to_string(row_ids[:i_start + 1])
                                gold = tokenizer.convert_tokens_to_string(row_ids[i_start + 1])
                                argmax_ids = row_rank[i_start + 1, :num_predictions].tolist()
                                assert len(argmax_ids) == num_predictions
                                argmax_tokens = tokenizer.convert_ids_to_tokens(argmax_ids)
                                candidates = [tokenizer.convert_tokens_to_string([tok]) for tok in argmax_tokens]

                                print('{} [{}] => {}'.format(chunk, gold, candidates))
                            print('')

            else:
                lm_loss = outputs[0]


    lst = []
    for x, y in zip(cache['loss'], cache['size']):
        lst += [x] * y

    eval_loss = np.mean(lst)
    bppl = torch.exp(torch.tensor(eval_loss))
    num_bytes = np.sum(cache['num_bytes'])

    result = {
        "loss": eval_loss,
        "bppl": bppl,
        "num_bytes": num_bytes,
    }

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    logger.info("***** Writing to {} *****".format(output_eval_file))
    os.system('mkdir -p {}'.format(os.path.dirname(output_eval_file)))
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    if args.write_xent:
        output_eval_file = os.path.join(eval_output_dir, prefix, "lm_xent.txt")
        logger.info("***** Writing to {} *****".format(output_eval_file))
        os.system('mkdir -p {}'.format(os.path.dirname(output_eval_file)))

        with open(output_eval_file, "w") as f:
            for wlst, vlst, rlst in zip(cache['gold'], cache['gold_prob'], cache['gold_rank']):
                for w, v, r in zip(wlst, vlst, rlst):
                    o = collections.OrderedDict()
                    o['tok'] = w
                    o['p'] = v
                    o['rank'] = r
                    f.write('{}\n'.format(json.dumps(o)))
                f.write('\n')

    return result


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--train_data_file", default=None, type=str, required=True,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")

    parser.add_argument('--use_template', action='store_true')
    parser.add_argument('--no_lookahead', action='store_true')
    parser.add_argument('--tree_mode', default='none', type=str)
    parser.add_argument('--show_predictions', action='store_true')
    parser.add_argument('--actions_file', default='actions.vocab', type=str)
    parser.add_argument('--write_xent', action='store_true')
    parser.add_argument('--partial', action='store_true')

    parser.add_argument("--model_type", default="bert", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--model_name_or_path", default="bert-base-cased", type=str,
                        help="The model checkpoint for weights initialization.")

    parser.add_argument("--mlm", action='store_true',
                        help="Train with masked-language modeling loss instead of language modeling.")
    parser.add_argument("--mlm_probability", type=float, default=0.15,
                        help="Ratio of tokens to mask for masked language modeling loss")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as model_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--block_size", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=5000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    assert args.block_size in (128, 256)
    my_globals['max_length'] = 40 if args.block_size == 256 else 20
    args.block_size = 512

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError("BERT and RoBERTa do not have LM heads but masked LM heads. They must be run using the --mlm "
                         "flag (masked language modeling).")
    if args.eval_data_file is None and args.do_eval:
        raise ValueError("Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
                         "or remove the --do_eval argument.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                    args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.block_size <= 0:
        args.block_size = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.block_size = min(args.block_size, tokenizer.max_len_single_sentence)
    model = model_class.from_pretrained(args.model_name_or_path,
                                        from_tf=bool('.ckpt' in args.model_name_or_path),
                                        config=config,
                                        cache_dir=args.cache_dir if args.cache_dir else None)
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        global_step, tr_loss = train(args, train_dataset, model, tokenizer)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)


    # Saving best-practices: if you use save_pretrained for the model and tokenizer, you can reload them using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))

        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
        model.to(args.device)


    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""

            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, tokenizer, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)

    return results


if __name__ == "__main__":
    main()
