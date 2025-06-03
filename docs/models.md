Model Zoo Information
=====================

Introduction
------------

Our Model Zoo is a curated collection of both open-source and closed-source large language models (LLMs) that are tested and benchmarked
to provide insights into their performance, environmental impact and alignment with municipal values.
As soon as new models are added to the Model Zoo and evaluated, they are ranked on a publicly available [leaderboard](https://amsterdam.github.io/grip-on-llms)
based on various aspects, including facutality, inclusivity, and carbon footprint.
This leaderboard helps users identify the strengths and weaknesses of each model, facilitating informed decisions based on specific needs and criteria.

Model Selection
---------------

We selected a diverse set of models from different companies, including both open-source and closed-source options. Our aim was to incorporate models whose licenses support both research and commercial purposes. While we are aware that with our current Model Zoo, direct comparisons are challenging due to varying model sizes, we opted for smaller versions of larger models to facilitate code development and testing. Once development is complete, we plan to incorporate larger versions if available.

Current Models in the Zoo
-------------------------

We currently support a number of open-source models hosted on HuggingFace (
[Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0),
[Mistral 7B Instruct v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3),
[Llama 3.1 8B Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct),
[Phi 4 Mini Instruct](https://huggingface.co/microsoft/Phi-4-mini-instruct),
[Falcon 3 7B Instruct](https://huggingface.co/tiiuae/Falcon3-7B-Instruct)
), as well as closed-source models securely hosted in our Azure environment (
[GPT-4o](https://openai.com/index/hello-gpt-4o/), 
[GPT-4o-mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)
).

This list is subject to updates as new models are added.


How to add models
-----------------
### For all models:
1. Check **compliance**:
   * Check whether the **license** allows us to use it following the guidelines below.
   * Check whether **training data** poses privacy or copyright concerns (guidelines to follow)
   * Check if the model **can be hosted or deployed** in our cloud environment
2. Add to the **list of supported models** in this docs page
3. Add to the **[leaderboard](/llm-eval-website/_data/models.json)** including information such as provider, costs, etc

### HuggingFace models:
1. **Add to [model_config.py](/llm_eval/language_models/llms/llm_config.py)**
   * add the full repo id
   * add any custom arguments if required
   * add a recommended system prompt if it exists
2. **Confirm apply_chat_template works**
   * Test whether it applies the expected model template or just a default template which might decrease performance.
3. **Test** if the model works (e.g. using the [test_llms.py](/tests/test_llms.py) file)


License Information
-------------------

The models in the Model Zoo are governed by various licenses, which dictate how they can be used, modified, and redistributed. Below is the license information for the models included as of April 15, 2025:

*   **[TII Falcon License](https://falconllm.tii.ae/falcon-terms-and-conditions.html) (Falcon3-7b-instruct):**
    *   Ensure compliance with the Acceptable Use Policy.
    *   Include necessary legal agreements and attribution statements when redistributing or publicly discussing derivative works.
*   **[Llama 3.1 Community License](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct/blob/main/LICENSE) (Llama-3.1-8b-instruct):**
    *   Display "Built with Llama" and include "Llama" in the name of any AI model created using the materials.
    *   Organizations with more than 700 million monthly active users must request a separate license from Meta.
*   **[Apache License 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md) (Mistral-7b-instruct-v3.0 and TinyLlama-1.1b-Chat-v1.0):**
    *   Provide a copy of the license with any distributed work or derivative works.
    *   Include notices of modifications and retain all copyright, patent, trademark, and attribution notices.
*   **[MIT License](https://huggingface.co/microsoft/Phi-4-mini-instruct/resolve/main/LICENSE) (phi-4-mini-instruct):**
    *   Include the original copyright notice and permission notice in all copies or substantial portions of the software.
    *   Understand that the software comes without any warranty, and assume any risks associated with its use.

Checklist for Model Usage
-------------------------

To make informed decisions about which models to use based on their licenses, consider the following checklist:
1.  **License Type:**
    *   Apache 2.0 and MIT licenses generally allow for broad use, but always review specific terms.
    *   Custom licenses require careful examination of terms and conditions.
2.  **Redistribution and Modification:**
    *   Ensure compliance with attribution and documentation requirements.
    *   Check for any restrictions on redistribution or modification.
3.  **Policy Compliance:**
    *   Regularly review updates to both external Acceptable Use Policies (AUPs) and internal organizational policies to ensure ethical and responsible use of AI technologies. AUPs are guidelines that define how technologies can be used, specifying permitted and and prohibited actions to ensure compliance with industry standards and ethical guidelines. This includes staying informed about industry standards and any changes to our organization's policies, particularly those concerning generative AI. As of April 18, 2025, the use of genAI models is not permitted under our organization's policies, but this may change in the future.
4.  **Legal Compliance:**
    *   Ensure that any use complies with applicable laws and regulations, including but not limited to:
        - **AI Act:** A regulatory framework proposed by the European Union to ensure safe and ethical use of AI technologies.
        - **GDPR (General Data Protection Regulation):** EU legislation focused on data protection and privacy, which impacts how AI models handle personal data.
