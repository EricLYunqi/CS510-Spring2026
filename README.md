## PageContext

Final report is `report.pdf`.

```
- baseline_api.py # Run baselines with commercial LLMs.
- baseline.py # Run baselines with open-sourced LLMs.
- page_context_api.py # Run PageContext with commercial LLMs.
- page_context.py # Run PageContext with open-sourced LLMs.
- preliminary_study.py # Run verification of frequency locality.
- draw.py # Draw figures in the report
```

Before you start, make sure to have a `.env` file which includes the Azure endpoints and API key for the following four LLMs:
- ChatGPT-5-mini
- Claude-Haiku-4.5
- DeepSeek-V4-Flash
- Kimi-K2.5

Dependenices are in `requirements.txt`, platform is RHEL and CUDA 12.8.

Also please git clone the benchmark:
```bash
https://github.com/snap-research/locomo.git
```

And create a `results` dir with the same layout in this repo. Currently it's only for demo does not contain the full evaluation results.