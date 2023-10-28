import copy
from functools import partial
from typing import Dict
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizer, DataCollatorForSeq2Seq
from datasets import load_dataset

from composer.core.data_spec import DataSpec
from composer.utils import dist
from llmfoundry.data.text_data import get_tokens_per_batch_func

CONTEXT_MARKUP_TOKENS = ["<paragraph>", "</paragraph>"]

PROMPT_DICT = {
    "prompt_input": (
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
    ),
    "prompt_no_input": (
        "### Instruction:\n{instruction}\n\n### Response:\n"
    ),
}

def build_self_rag_dataloader(
    cfg: DictConfig,
    tokenizer: PreTrainedTokenizerBase,
    device_batch_size: int) -> DataSpec:

    assert tokenizer.pad_token is not None

    context_markups = []
    for token in ["<paragraph>", "</paragraph>"]:
        context_markups.append(tokenizer.convert_tokens_to_ids(token))

    # load dataset
    dataset = load_dataset(cfg.dataset.hf_name, split=cfg.dataset.split)
    
    # preprocess dataset
    encode_function = partial(
        encode_with_prompt_completion_format,
        tokenizer=tokenizer,
        max_seq_length=cfg.max_seq_length,
        context_markups=context_markups
    )

    dataset = dataset.map(
        encode_function,
        batched=False,
        num_proc=32,
        remove_columns=[name for name in dataset["train"].column_names if name not in ["input_ids", "labels", "attention_mask"]],
        desc="Tokenizing and reformatting instruction data",
    )

    # st
    dl = DataLoader(
        dataset,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding="longest"),
        batch_size=device_batch_size,
        drop_last=cfg.drop_last,
        sampler=dist.get_sampler(dataset,
                drop_last=cfg.drop_last,
                shuffle=cfg.dataset.shuffle),
        num_workers=cfg.num_workers,
        pin_memory=cfg.get('pin_memory', True),
        prefetch_factor=cfg.get('prefetch_factor', 2),
        persistent_workers=cfg.get('persistent_workers', True),
        timeout=cfg.get('timeout', 0),
    )

    token_counting_func = get_tokens_per_batch_func(
        pad_token_id=tokenizer.pad_token_id)
    
    return DataSpec(dataloader=dl, get_num_tokens_in_batch=token_counting_func)

def encode_with_prompt_completion_format(example, tokenizer, max_seq_length, context_markups=None):
    '''
    Here we assume each example has 'prompt' and 'completion' fields.
    We concatenate prompt and completion and tokenize them together because otherwise prompt will be padded/trancated
    and it doesn't make sense to follow directly with the completion.
    '''
    # if prompt doesn't end with space and completion doesn't start with space, add space

    prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
    source_text = prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
    target_text = example['output'] + tokenizer.eos_token
    examples_tokenized = _tokenize_fn(source_text + target_text, tokenizer, max_seq_length)
    sources_tokenized = _tokenize_fn(source_text, tokenizer, max_seq_length)

    input_ids = examples_tokenized["input_ids"].flatten()
    source_len = sources_tokenized["input_ids_lens"]
    labels = copy.deepcopy(input_ids)
    labels[ :source_len-1] = -100

    if context_markups is not None:
        context_start = False
        for j, orig_token in enumerate(labels[source_len:]):
            if context_start is False and orig_token == context_markups[0]:
                context_start = True
                assert labels[source_len+j] == context_markups[0]
                start_idx = j+source_len
                end_idx = None
                for k, orig_token_2 in enumerate(labels[start_idx:]):
                    if orig_token_2 == context_markups[1]:
                        end_idx = start_idx + k
                if end_idx is None:
                    end_idx =  start_idx + k
                else:
                    assert labels[end_idx] == context_markups[1]
                labels[start_idx+1:end_idx] = -100
                context_start = False
    attention_mask = torch.ones_like(input_ids)

    return {
        'input_ids': input_ids.flatten(),
        'labels': labels.flatten(),
        'attention_mask': attention_mask.flatten()
    }

def _tokenize_fn(text: str, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Dict:
    """Tokenize a list of strings."""
    input_ids = labels = tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=max_seq_length,
            truncation=True,
    ).input_ids
    input_ids_lens = labels_lens = input_ids.ne(tokenizer.pad_token_id).sum().item()

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )