System Prompt Experiments
=========================

Introduction
------------
Based on the Grip on LLMs research so far, we have identified a number of aspects
which experts find important, as well as specific requirements related to them.
These insights can be used not only to test the models' performance,
but they can also be turned around as explicit instructions for the LLMs in order to
steer them into generating responses aligned with our standards and expectations.


Values
------
When looking at aspects and values which need to be included, we considered the following:
- [Amsterdam's Vision and Agenda for AI](https://www.amsterdam.nl/innovatie/amsterdamse-visie-ai/)
    according to which AI developments in the city should be _**human-centric**_, _**reliable**_ and _**future-proof**_.
- [the TADA manifest](https://openresearch.amsterdam/en/page/110506/tada.city),
    which states that data-driven systems should, among others, be _**transparent**_,
    contribute to _**inclusivity**_ in our city and _**never have the last word**_.
- [the Grip on LLMs aspects](/docs/aspects.md), including _**factuality**_, _**honesty**_, _**sustainability**_ and others


Next, we wrote out concrete instructions which reflect these aspects and describe the expected behavior.
For example, to ensure [honesty](/docs/benchmarks_honesty.md), we used the
[category definitions](/llm_eval/benchmarks/honesty/honesty_categories.py)
which we had initially used to evaluate the base model performance on this aspect.

Evaluation
----------
To test the performance of prompt and make some decisions about its content and structure
we used the existing benchmarks for relevant aspects - honesty and bias.
We ran experiment with and without a system prompt, and with variations of the prompt.

Disclaimer: We have not done extensive optimization of the prompt.
Decisions were mostly related to the language of the prompt and its overall level of detail.


Findings: Language
------------------
One of the most important decisions we had to make was related to the language of the system prompt.
#TODO: Fill in


Findings: Honesty
-----------------
#TODO: Fill in
Remarks:
* Plot improvements

Findings: Bias
--------------
#TODO: Fill in
Remarks:
* Plot improvements


Findings: Model Family
----------------------
#TODO: Fill in
Remarks:
* tested only for some models (mostly for Mistral & GPT)
* long prompt might cause issues for smaller window models
*
