# Model Costs

---

## Why publish cost‑per‑prompt?

Listing a single *cost‑per‑prompt* number lets you compare:

* closed‑source API models that charge per token, and
* self‑hosted models that run on our Azure infrastructure

Moreover, it is more intuitive and easier to understand than price per million tokens.

---
## Data for Comparison:
We utilize the summarization and text simplification datasets to measure the price as it is open-ended questions. This way we also take into account whether models tend to write longer answers than others.
We explicitly exclude multiple-choice tasks.


---

## How we calculate the number you see

| Model type                           | Formula we apply                                               | How we calculate                                                                                                                                   |
|--------------------------------------|----------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| **API model** (Azure OpenAI Endpoint) | input tokens × *vendor price* + output tokens × *vendor price* | We estimate the number of tokens using [Tiktoken](https://github.com/openai/tiktoken?tab=readme-ov-file#-tiktoken). The input and output tokens are the average amount of input and outputs tokens utilized by an LLM.       |
| **Self-hosted model** (In Azure AML)       | average prompt duration × *GPU hourly price*                     | We only measure the cost during inference, the VM warmup time and loading model time are excluded. We calcultate the average duration time of a prompt. We run all experiments on an H100 compute, making the run times comparable, although we realize that different compute resources might be more suitable for bigger or smaller models. |

The Azure OpenAI vendor prices are taken from the official [pricing page](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/).
The GPU hourly rates are taken from our secure environment where they are published in US dollars. We apply the conversion rate *1 USD = 0.8679 EUR* mentioned on [the Azure website](https://azure.microsoft.com/nl-nl/pricing/details/cognitive-services/openai-service/) to convert these rates into euros. All prices and rates are from 26th August 2025.

---

## Caveats & future work

* **Prices change**: Vendor prices and exchange rates are regularly updated and quickly become outdated.
* **Region**: All costs assume GPU deployments in *West Europe* and model deployments in *Sweden*. Prices might differ for other regions.
* **Spot instances**: For open‑source models we show on‑demand rates only; spot pricing can be cheaper but is not guaranteed.
