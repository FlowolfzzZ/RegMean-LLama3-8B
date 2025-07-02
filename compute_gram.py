from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline
from transformers import BitsAndBytesConfig
from torch import nn
import re
from tqdm import tqdm
import torch
import os
import pickle
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator, DataCollatorWithPadding
import argparse

def tokenize_function(example, tokenizer):
    input = tokenizer.apply_chat_template([
        {'role':'system', 'content': example['instruction']},
        {'role':'user', 'content': example['input']},
        {'role':'assistant', 'content': example['output']}
    ], tokenize=False)
    return tokenizer(
        input,
        truncation=True,
        max_length=256,
        padding=True,
        padding_side='left'
    )

def filter_params_to_merge(param_names, exclude_param_regex):
    params_to_merge = []
    for name in param_names:
        valid = not any([re.match(patt, name) for patt in exclude_param_regex])
        if valid:
            params_to_merge.append(name)
    return params_to_merge


def filter_modules_by_regex(base_module, include_patterns, include_type):
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any([re.match(patt, name) for patt in include_patterns])
        valid_type = not include_type or any([isinstance(module, md_cls) for md_cls in include_type])
        if valid_type and valid_name:
            modules[name] = module
    return modules

def compute_gram(model, train_dataloader, n_step, handles):
    grams = {} # gram matrices for each linear layer inputs
    xn = {} # number of examples used for computing gram

    def get_gram(name):
        def hook(module, input, output):
            #print(input[0].shape)
            x = input[0].detach().float() # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0,1), x) # [h,h]

            if name not in grams:
                raw_val = xtx / x.size(0)
                grams[name] = raw_val.cpu()
                xn[name] = x.size(0)
            else:
                past_gram = grams[name].cuda()
                raw_val = (past_gram * xn[name] + xtx) / (x.size(0) + xn[name])
                grams[name] = raw_val.cpu()
                xn[name] += x.size(0)
        return hook

    linear_modules = filter_modules_by_regex(model, None, [nn.Linear])

    print('Identified linear layers')
    for k in linear_modules:
        print(k)
    
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_gram(name))
        handles.append(handle)

    total = n_step if n_step > 0 else len(train_dataloader)
    for step, inputs in tqdm(enumerate(train_dataloader), total=total, desc='Computing gram matrix'):
        if n_step > 0 and step == n_step:
            break
        inputs ={k: v.cuda() for k,v in inputs.items()}
        outputs = model(**inputs)

    for handle in handles:
        handle.remove()

    return grams

def clear_hooks(modules):
    for n, module in modules.items():
        if 0 in module._forward_hooks:
            module._forward_hooks.pop(0)

def save_grams(grams, root):
    with open(os.path.join(root, 'grams.pkl'),'wb') as wf:
        state = {k: v.cpu() for k,v in grams.items()}
        pickle.dump(state, wf)

def load_grams(root):
    with open(os.path.join(root, 'grams.pkl'),'rb') as f:
        obj = pickle.load(f)
    return obj

def main(
    model_name: str,
    dataset_name: str,
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    # train_ds = load_dataset(dataset_name, split='train[:10000]')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # quant_config = BitsAndBytesConfig(load_in_8bit=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    
    train_ds = load_dataset("json", data_files=dataset_name, split='train[:2000]')
    tokenized_ds = train_ds.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer})
    tokenized_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(tokenized_ds, batch_size=32, collate_fn=collator)

    handles = []
    with torch.no_grad():
        grams = compute_gram(model, train_loader,  -1, handles)

    with open(os.path.join(output_dir, 'gram.pkl'),'wb') as wf:
        pickle.dump(grams, wf)

if __name__ == '__main__':
    import fire
    fire.Fire(main)