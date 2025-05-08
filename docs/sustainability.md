Sustainability Information
==========================

Introduction
------------

This documentation provides guidelines on using [CodeCarbon](https://codecarbon.io/) to measure CO2-equivalent emissions and energy usage for benchmarking large language models (LLMs). The goal of using CodeCarbon in this project is to accurately measure and compare the CO2-equivalent emissions and energy usage of different large language models (LLMs) during benchmarking. This helps assess their environmental impact and make informed decisions about model selection based on sustainability criteria. Due to limited information about the Azure Cloud environment, we utilize the OfflineEmissionsTracker instead of the online version (EmissionsTracker). 

Please note that certain LLMs, such as OpenAI's GPT models, were not included in the CodeCarbon assessment. Due to the lack of transparency regarding energy usage and other relevant data when using their API, it is not possible to make an accurate environmental impact assessment, resulting in null values for these models.

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
        
    

Measuring CO2-eq and Energy Usage
---------------------------------

### Step-by-Step Guide

1.  **Initialize the Emissions Tracker:** Create an instance of the `OfflineEmissionsTracker` at the beginning of your benchmark script:
    
        tracker = OfflineEmissionsTracker(country_iso_code="SE")
        
    
2.  **Start Tracking:** Begin tracking emissions and energy usage before running your benchmarks:
    
        tracker.start()
        
    
3.  **Run Benchmark Tests:** Execute your benchmark tests for each LLM. Ensure that the code for running the models is encapsulated between the start and stop tracking commands.
    
4.  **Stop Tracking:** After the benchmark run completes, stop the tracker to record the emissions data:
    
        tracker.stop()
        
    
5.  **Retrieve Results:** CodeCarbon will automatically log the CO2-eq emissions in kilograms and energy usage. You can access these logs to compare the environmental impact of each LLM. Obtain the CodeCarbon emissions logs as a dictionary:

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

Conclusion
----------

By integrating CodeCarbon's (Offline)EmissionsTracker into your benchmarking process, you can effectively measure and compare the environmental impact of various large language models. This documentation serves as a basic guide to get started with tracking emissions and energy usage.

* * *