# Grip on LLMs


While Large Language Models (LLMs) provide significant opportunities
for governmental organization to automate processes and improve services for citizens,
there is limited research and knowledge into the level to which
current models reflect the ethical standards and cultural nuances of Dutch society.

Our mission is to empower the City of Amsterdam and inspire the Dutch government
to responsibly implement LLMs by creating an overview of their
performance, environmental impact and alignment with human values.

This repository contains code and methodologies related to this evaluation.

## Background
WIP


## Project Structure

* [`llm_eval`](./llm_eval)
  * [`llm_eval/benchmarks`](./llm_eval/benchmarks) - 
    contains a base benchmark class as well as the currently supported benchmarks.
  * [`llm_eval/language_models`](./llm_eval/language_models) - 
    enables the use of diverse language models by different providers.
    Currently, we support Azure deployments of GPT as well as HuggingFace models.
* [`tests`](./tests) - tests the usage of models and benchmarks.

## Installation & Setup

1) Clone this repository:

```bash
git clone https://github.com/Amsterdam/grip-on-llms.git
```

2) Install poetry if needed by following the [instructions](https://python-poetry.org/docs/).

3) Navigate to project directory and install all dependencies:

```bash
poetry install
```
4) Set up pre-commit hooks
```bash
poetry run pre-commit install
```

5) Optional: Change HuggingFace cache (e.g. to the shared storage account folder)
When using HuggingFace model, evaluation metrics, etc, all necessary artefacts
are cached in a default directory. To change this directory, you can set the
environment variable `HF_HOME` (export prior to running experiments or set
during run time using `os.environ["HF_HOME"] = {folder}`.

6) Optional: Manually set CPU power TDP value if requested.

When running CodeCarbon for the first time, the output will display hardware information. Occasionally, an unknown CPU error may occur. If this happens, manually add the CPU and its TDP value, which can be found online, to the cpu_power.csv file located at: /anaconda/envs/{name environment}/lib/{python version}/site-packages/codecarbon/data/hardware/cpu_power.csv.

```bash
vim /anaconda/envs/{name environment}/lib/{python version}/site-packages/codecarbon/data/hardware/cpu_power.csv
```

The code has been tested with Python 3.9 on Linux.

## Contributing

Feel free to help out! [Open an issue](https://github.com/Amsterdam/grip-on-llms/issues) or submit a [PR](https://github.com/Amsterdam/grip-on-llms/pulls).


## Acknowledgements

This repository was created by the **_Grip op LLMs_** team for the City of Amsterdam.


## License 

This project is licensed under the terms of the European Union Public License 1.2 (EUPL-1.2).
