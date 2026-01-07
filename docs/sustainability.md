Sustainability Information
==========================

Introduction
------------

This documentation provides guidelines on using [CodeCarbon](https://codecarbon.io/) to measure CO2-equivalent emissions and energy usage for benchmarking large language models (LLMs). The goal of using CodeCarbon in this project is to accurately measure and compare the CO2-equivalent emissions and energy usage of different large language models (LLMs) during benchmarking. This helps assess their environmental impact and make informed decisions about model selection based on sustainability criteria. Due to limited information about the Azure Cloud environment, we utilize the OfflineEmissionsTracker instead of the online version (EmissionsTracker).

Please note that certain LLMs, such as OpenAI's GPT models, were not included in the CodeCarbon assessment. Due to the lack of transparency regarding energy usage and other relevant data when using their API, it is not possible to make an accurate environmental impact assessment, resulting in null values for these models.

We measure energy consumption (e.g., in kilowatt-hours) rather than carbon emissions to ensure consistency and comparability across benchmarks [1]. Carbon emissions can vary significantly depending on the carbon intensity of the energy grid at different physical locations. By focusing on energy consumption, we eliminate these location-based discrepancies and provide a more objective measure of efficiency.

Additionally, we have not yet worked on the interpretability of the results obtained from these assessments. However, we plan to address this and incorporate interpretability features into the analysis by mid-2025. Furthermore, we aim to visualize the results using a dedicated dashboard to provide a clearer and more interactive representation of the environmental impact data.

Prerequisites
-------------

*   Python installed on your system
*   CodeCarbon library installed (`pip install codecarbon`)
*   Access to the LLMs you wish to benchmark

Setup
-----

1.  **Install CodeCarbon:** Ensure CodeCarbon is installed in your Python environment:

        pip install codecarbon


2.  **Import CodeCarbon:** In your Python script, import the necessary module:

        from codecarbon import OfflineEmissionsTracker



Measuring Energy Usage
----------------------

### Step-by-Step Guide

1.  **Initialize the Emissions Tracker:** Create an instance of the `OfflineEmissionsTracker` at the beginning of your benchmark script:

        tracker = OfflineEmissionsTracker(country_iso_code="SE")


2.  **Start Tracking:** Begin tracking emissions and energy usage before running your benchmarks:

        tracker.start()


3.  **Run Benchmark Tests:** Execute your benchmark tests for each LLM. Ensure that the code for running the models is encapsulated between the start and stop tracking commands.

4.  **Stop Tracking:** After the benchmark run completes, stop the tracker to record the emissions data:

        tracker.stop()


5.  **Retrieve Results:** CodeCarbon will automatically log the CO2-eq emissions in kilograms and energy usage. We use the energy use value to compare LLMs. You can access these logs to compare the environmental impact of each LLM. Obtain the CodeCarbon emissions logs as a dictionary:

        final_results = tracker.final_emissions_data.__dict__

Results Description
------------------

The output from CodeCarbon provides detailed insights into the environmental impact and energy usage of each benchmark run. It includes information such as:
*   **Timestamp:** Records the date and time of the benchmark.
*   **Project and Run Identifiers:** Unique identifiers for the project and specific benchmark run.
*   **Duration:** The time taken for the benchmark run.
*   **Emissions Data:** CO2-equivalent emissions and emissions rate.
*   **Power and Energy Usage:** Metrics for CPU, GPU, and RAM power consumption and energy usage.
*   **Total Energy Consumption:** Overall energy used during the benchmark.
*   **Location and Environment Details:** Information about the geographical location and cloud environment.
*   **System Specifications:** Details about the operating system, Python version, and hardware used.
*   **Tracking Mode and Efficiency:** Includes tracking mode and Power Usage Effectiveness (PUE).
This output enables comprehensive analysis and comparison of the environmental impact of different LLMs, supporting informed decisions based on sustainability criteria.

Analyzing Results
-----------------

*   **Log Files:** CodeCarbon generates log files containing detailed information about CO2-eq emissions and energy usage.
*   **Comparison:** Use these logs to compare the environmental impact of different LLMs based on their CO2-eq emissions and energy consumption.

Evaluation Metrics
-----------------------

#TODO: (long story short: nothing works, but we use SARI for now)

### Mapping to Categories

Finally, we describe our methodology for mapping the raw scores from the benchmarks to the categories visualized in our [LLM Overview](https://amsterdam.github.io/grip-on-llms).

We currently calculate average energy use per benchmark. This is based on the total energy usage across all prompts in a benchmark (e.g. 100 prompts) and then averages it. The normalized energy usage is then categorized into five levels:


|           | Average Energy use per benchmark (kWh)     | Level     |
|-----------|:------------------|:----------|
| <img src="https://readme-swatches.vercel.app/EC0000?style=circle" width="20" height="20" alt="Red Circle"> | >0.1   | Very High   |
| <img src="https://readme-swatches.vercel.app/FF9100?style=circle" width="20" height="20" alt="Orange Circle"> | 0.05-0.1  | High        |
| <img src="https://readme-swatches.vercel.app/FFE600?style=circle" width="20" height="20" alt="Yellow Circle"> | 0.025-0.05   | Medium     |
| <img src="https://readme-swatches.vercel.app/BED200?style=circle" width="20" height="20" alt="Lime Circle"> | 0.015-0.025   | Low       |
| <img src="https://readme-swatches.vercel.app/00A03C?style=circle" width="20" height="20" alt="Green Circle"> | 0-0.015   | Very Low  |

### Future Considerations

If a benchmark has significantly more prompts (e.g. 1000 prompts), it will disproportinately affect the average energy use. This makes comparisons between benchmarks inconsistent. In the near future, we will therefore divide the total energy usage by the number of prompts in the benchmarks before calculating the mean. We may also adjust the unit of measurement for readability purposes.

Conclusion
----------

By integrating CodeCarbon's (Offline)EmissionsTracker into your benchmarking process, you can effectively measure and compare the environmental impact of various large language models. This documentation serves as a basic guide to get started with tracking emissions and energy usage.

References
----------

- [1]  ["AI Energy Score Leaderboard Documentation"](https://huggingface.github.io/AIEnergyScore/#disclosing-results).

* * *
