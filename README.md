# Data Driven Forecasting Workflow

A workflow for streamflow forecasting for rivers across North America, based on streamflow and snow water equivalent station observations.

## Description

This repository contains a reproducible data-driven workflow, applied to numerous basins, leveraging snow water equivalent (SWE) measurements as predictors and streamflow observations as predictands from datasets like CanSWE, NRCS, SNOTEL, HYDAT, and USGS. Gap filling for SWE datasets is done using quantile mapping from nearby stations and Principal Component Analysis to identify independent predictor components. These components are employed in a regression model to generate ensemble hindcasts of streamflow volumes for 75 nival basins with minimal regulation from 1979 to 2021. This work focuses on a user-oriented hindcast evaluation and provides insights beneficial to snow surveyors, forecasters, workflow developers, and decision-makers.

## Repository Structure

- 📂 `notebooks/`: Collection of Jupyter Notebooks detailing the forecasting workflow.
- 📂 `scripts/`: Functions used in data processing and analysis
- 📂 `settings/`: Settings for running the forecasting workflow
- 📂 `test_cases/`: Sample data for running the forecasting workflow for a single basin
- 📄 `requirements.txt`: Lists the Python packages required for reproducing the workflow.

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/lou-a/data_driven_forecasting_workflow.git
   ```

2. **Set Up Virtual Environment (Optional)**  
   ```bash
   python -m venv env
   source env/bin/activate  # For Windows, use `env\Scripts\activate`
   ```

3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

4. **Navigate to the Notebooks Directory**  
   ```bash
   cd notebooks/
   ```

5. **Start Jupyter Notebook**  
   ```bash
   jupyter notebook
   ```

## Contribution

Feel free to fork this repository and contribute. For major changes, please open an issue first to discuss your proposed changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Citation

If you use this workflow or any data therein, please consider citing our paper. The citation details can be found in the `CITATION.md` file.
To be added ..

## Contact

For any inquiries or feedback, please open an issue or contact the maintainers via email.
