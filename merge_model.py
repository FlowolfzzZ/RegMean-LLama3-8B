from transformers import AutoModelForCausalLM
import argparse
import re
import torch
import pickle

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

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def avg_merge(local_models, regmean_grams=None, skip_all_regmean=False, regmean_alpha=0.9):
    params = {}
    for local_model in local_models:
        n2p = {k: v for k,v in local_model.named_parameters()}
        merge_param_names = filter_params_to_merge([n for n in n2p], ['.*classifier.*']) # for glue label spaces are different
        for n in merge_param_names:
            if n not in params:
                params[n] = []
            params[n].append(n2p[n])

    if regmean_grams: # regmean average
        avg_params = regmean_merge(params, regmean_grams, skip_all_regmean, regmean_alpha)

    else: # simple average
        avg_params = {k: torch.stack(v,0).mean(0) for k, v in params.items()}

    return avg_params

def copy_params_to_model(avg_params, model):
    for n, p in model.named_parameters():
        if n in avg_params:
            p.data.copy_(avg_params[n])

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret

def regmean_merge(all_params, all_grams, skip_all, regmean_alpha):
    avg_params = {}
    n_model = len(all_grams)
    for name in all_params:
        h_avged = False

        if not skip_all and name.endswith('.weight'):

            module_name = name[:-len('.weight')]
            if module_name in all_grams[0]: #and all_grams[0][module_name].shape[1] < dim_thres:
                
                print(f'Regmean: {name}')
                gram_m_ws, grams = [], []

                for model_id, model_grams in enumerate(all_grams):
                    # move to cuda here
                    param_grams = model_grams[module_name]
                    param = all_params[name][model_id].float() # to float32

                    param_grams = param_grams.cuda()
                    param = param.cuda()
                    
                    param_grams = reduce_non_diag(param_grams, a=regmean_alpha)
                    
                    gram_m_ws.append(torch.matmul(param_grams, param.transpose(0,1)))
                    grams.append(param_grams)

                sum_gram = sum(grams)
                sum_gram_m_ws = sum(gram_m_ws)
                
                # Update: solve least square regression directly for numerical stability

                #sum_gram_inv = torch.inverse(sum_gram) 
                #wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)

                wt = torch.linalg.lstsq(sum_gram, sum_gram_m_ws).solution
                
                w = wt.transpose(0,1)
                avg_params[name] = w.cpu()
                h_avged = True
        if not h_avged: # if not averaged with regmean, then do simple avg
            avg_params[name] = torch.stack(all_params[name],0).mean(0)
           
    return avg_params

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['avg','regmean'])
    args = parser.parse_args()


    model_name = 'rombodawg/rombos_Replete-Coder-Llama3-8B'
    model_code = AutoModelForCausalLM.from_pretrained(model_name)
    model_name = 'TIGER-Lab/MAmmoTH2-8B'
    model_math = AutoModelForCausalLM.from_pretrained(model_name)

    print('Loading code gram')
    with open('runs/merges/code-llama3/gram.pkl','rb') as f:
        code_gram = pickle.load(f)

    print('Loading math gram')
    with open('runs/merges/math-llama3/gram.pkl','rb') as f:
        math_gram = pickle.load(f)

    alpha = 0.1
    with torch.no_grad():
        regmean_avg_params = avg_merge([model_code, model_math],  regmean_grams=[code_gram, math_gram], regmean_alpha=alpha, 
                                       skip_all_regmean=args.method == 'avg')

    
    copy_params_to_model(regmean_avg_params, model_math) # merged param copied into this

    if args.method == 'regmean':
        output_dir = f'runs/merges/regmean_{alpha}/'
    else:
        output_dir = f'runs/merges/avg'
    
    model_math.save_pretrained(output_dir)



    