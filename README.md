# FROST BYTE: Forecasting River Outlooks from Snow Timeseries: Building Yearly Targeted Ensembles

FROST BYTE is a reproducible data-driven workflow for probabilistic seasonal streamflow forecasting, based on streamflow and snow water equivalent station observations.

## Description

This repository contains a reproducible data-driven workflow, leveraging snow water equivalent (SWE) measurements as predictors and streamflow observations as predictands, drawn from reliable datasets like CanSWE, NRCS, SNOTEL, HYDAT, and USGS. Gap filling for SWE datasets is done using quantile mapping from nearby stations and Principal Component Analysis is used to identify independent predictor components. These components are employed in a regression model to generate ensemble hindcasts of seasonal streamflow volumes. This workflow was applied by Arnal et al. to 75 river basins with a nival (i.e., snowmelt-driven) regime and with minimal regulation across Canada and the USA, for generating hindcasts from 1979 to 2021. This study presented a user-oriented hindcast evaluation, offering valuable insights for snow surveyors, forecasters, workflow developers, and decision-makers.

## Repository Structure

- 📂 `notebooks/`: Collection of Jupyter Notebooks detailing each step of the forecasting workflow.
- 📂 `scripts/`: Functions used in the data processing and analyses carried out in the Notebooks.
- 📂 `settings/`: Settings for running the forecasting workflow.
- 📂 `test_case_data/`: Sample data for running the forecasting workflow for two single river basins: the Bow River at Banff in Alberta, Canada, and the Crystal River Abv Avalanche Crk, Near Redstone in Colorado, USA.
- 📄 `requirements.txt`: Lists the Python packages required for reproducing the workflow.

## Getting Started

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/lou-a/FROST-BYTE.git
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
To be added ...

## Contact

For any inquiries or feedback, please open an issue or contact the maintainers via email.
