### Experiments Setup

- Benchmark: locomo.
- Evaluation metric: F1 score.
- Methods:
  - `baseline.py`: retrieval only.
  - `page_context.py`: maintain hot-accessed turns + retrieval.
- Platform: Google Colab.

### How to Run

It's best to run in notebook on Colab, but to run locally:

```bash
python baseline.py
python page_context.py
```

Important parameters:
```python
# TOP_K is the retrieved # of turns in baseline
TOP_K = 18 
# HOT_SIZE is the # of turns maintained persistently inside the memory, that is, the # of retrieved turns for page_context.py is 18 - 4 = 14
HOT_SIZE = 4
```

After running, please save the results (by directly copying the output cell from Colab) under `results/eval` with the format as `<method_name>_<model_name>.out`, e.g., `pagecontext_Qwen3_4B.output`. You can also add the saved `.json` file.

Also, the current two output files are only based on conversation `Conv_26`, but locomo has 10 such conversations. 

### Sprint

🏃🏼: Running by xxx.

- [] Qween3-4B (🏃🏼 / Yunqi)
- [] Qwen3-0.6B
- [] Qwen3-0.1.7B
- [] Qwen3-0.8B