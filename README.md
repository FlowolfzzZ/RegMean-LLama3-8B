## Fast and Numerically Stable RegMean for Merging LLama3-8B

This repo is a fast and numerically stable re-implementation of RegMean model merging algorithm for LLama3-8B.

We merge the following two models.

- [Code Model] [rombodawg/rombos_Replete-Coder-Llama3-8B](https://huggingface.co/rombodawg/rombos_Replete-Coder-Llama3-8B) (Re-implementation of Replete-Coder) 
- [Math Model][ TIGER-Lab/MAmmoTH2-8B](https://huggingface.co/TIGER-Lab/MAmmoTH2-8B)


## Results


| Method/Benchmark | GSM8k (Math) | Mathqa (Math) | HumanEval-Instruct (Code) | MBPP (Code) |
|  ---- | ---- | ---- | ---- | ---- |
|   |  5-shot EM  | 0-shot Acc-norm | 0-shot Pass@1 | 3-shot Pass@1 |
|  [Math Model](https://huggingface.co/TIGER-Lab/MAmmoTH2-8B) |  70.40* | 43.85 | 36.59 | 40.04 |
|  [Code Model](https://huggingface.co/rombodawg/rombos_Replete-Coder-Llama3-8B) | 57.92 | 37.35 | 42.07 | 49.20 |
|  [Average](https://huggingface.co/aucson/llama3-code-math-avg-merge) | 65.27 | 44.05 | 43.29  | 47.80 | 
|  [RegMean ($\alpha$=0.1)](https://huggingface.co/aucson/llama3-code-math-regmean-merge/tree/main) | 68.31 | 44.99 | 44.51 | 45.20 |

\* Official result

\* We found the zero-shot results are sensitive to chat templates and reported best achievable result for HumanInstruct for all models: we modified `lm-evaluation-harness/lm_eval/tasks/humaneval/humaneval.yaml` so that "\`\`\`" can be considered as end of responses.

The merged models, along with the activation inner product matrices, are avaiable on the huggingface hub.


## What's new?

RegMean solves a least square regression problem at each linear layer of the transformer. This is now implemented with built-in PyTorch linalg.lstsq function. 

```python
# old
# sum_gram_inv = torch.inverse(sum_gram) 
# wt = torch.matmul(sum_gram_inv, sum_gram_m_ws)

# new
wt = torch.linalg.lstsq(sum_gram, sum_gram_m_ws).solution
```

According to PyTorch's official doumentation,
```
This function computes X = A.pinverse() @ B in a faster and more numerically stable way than performing the computations separately.
```


## Computational efficiency

- **Computing gram matrices**: We compute inner product matrics for code and math models on 10k training examples. Each of them take 3-hour on one Quadro RTX A6000 GPU (which can probably accelerated with more efficient LLM inference framework). But we have uploaded them under the [merged model repo](https://huggingface.co/aucson/llama3-code-math-regmean-merge/tree/main) so that you do not need to re-compute.

- **Merging Models**: ~2 minutes on the same GPU for this re-implementation. Please note loading two 8B models and (almost) equally sized inner product matrices at once can take >150GB memory.

## Reproducing the results

1. Create a python environment and install the modified lm-eval-harness library for evaluating merged models.

```
cd lm-eval-harness
pip install -e .
```
The only modification is `lm_eval/tasks/humaneval/humaneval.yaml`.

2. Preparing activation inner product matrices.

You can download them from the [merged model repo](https://huggingface.co/aucson/llama3-code-math-regmean-merge/tree/main) and place them under `runs/merges/math-llama3/gram.pkl` and `runs/merges/code-llama3/gram.pkl`. Alternatively, you can compute them yourself with,

```
python compute_gram.py code
python compute_gram.py math
```

3. Merging models

```
python merge_model.py avg
python merge_model.py regmean

```

4. Evaluation with `lm-eval-harness`. Please follow the safety guidelines of humaneval and mbpp regarding execution of LLM generated code.

```
merge_exp=regmean_0.1 
# merge_exp=avg

HF_ALLOW_CODE_EVAL=1 lm_eval --model vllm --model_args pretrained=runs/merges/${merge_exp},tokenizer=meta-llama/Meta-Llama-3-8B,tensor_parallel_size=1,dtype=bfloat16 --tasks mathqa,gsm8k,humaneval_instruct,mbpp --output_path runs/merges/${merge_exp}/lm_eval_results_preds --log_samples --trust_remote_code --confirm_run_unsafe_code
```

## Caveats

Overall, simple averaging works well for LLMs and the benefits of merging algorithms diminishes for merging algorithms [1]


## Citations

For the RegMean algorithm.
```
@inproceedings{
    jin2023dataless,
    title={Dataless Knowledge Fusion by Merging Weights of Language Models},
    author={Xisen Jin and Xiang Ren and Daniel Preotiuc-Pietro and Pengxiang Cheng},
    booktitle={The Eleventh International Conference on Learning Representations },
    year={2023},
    url={https://openreview.net/forum?id=FCnohuR6AnM}
}
```

Here are other useful references that greatly inspired this re-implementation.

[1] Yadav et al. 2024, [What Matters for Model Merging at Scale?](https://arxiv.org/abs/2410.03617)

[2] Tam et al. 2024, [Merging by Matching Models in Task Parameter Subspaces](https://openreview.net/forum?id=qNGo6ghWFB)
