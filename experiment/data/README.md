# Experimental Data Documentation

---

## Dataset Overview

This document describes the experimental data used in this project, primarily focusing on **epidemiological statistics**.

---

## Data Files

The following data file is used and can be loaded via the `data_loader` script (referencing `"data/epidemiology.csv"`).

| File Name | Description | Source Link |
| :--- | :--- | :--- |
| **`epidemiology.csv`** | Contains country-level epidemiological data (e.g., cases, deaths, hospitalizations). | [Google COVID-19 Open Data - `table-epidemiology.md`](https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-epidemiology.md) |

---

## How to Obtain the Data

To run the project, please download the required `epidemiology.csv` file from the official Google Cloud Platform COVID-19 Open Data repository:

* **Direct Source:** [Download `epidemiology.csv` (View Documentation)](https://github.com/GoogleCloudPlatform/covid-19-open-data/blob/main/docs/table-epidemiology.md)

**Note:** Ensure you place the downloaded file into the expected path, which is typically `data/`.

---

## Country Code Converter

The project also uses a separate file for converting country codes:

* **`alpha2_country.csv`**: This file is used internally by the project for converting between different country code standards (e.g., Alpha-2 codes to country names). This is a standard converter file and is typically included with the project source.