# FROST BYTE: Forecasting River Outlooks from Snow Timeseries: Building Yearly Targeted Ensembles
[![License: GPL v3](https://img.shields.io/badge/License-GPL%20v3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/lou-a/FROST-BYTE/pulls)
[![Twitter](https://img.shields.io/twitter/url/https/twitter.com/cloudposse.svg?style=social&label=Follow%20%40ArnalLouise)](https://twitter.com/ArnalLouise)




FROST BYTE is a reproducible data-driven workflow for probabilistic seasonal streamflow forecasting, based on streamflow and snow water equivalent station observations.

<p align="center">
<img src="FROST BYTE logo.png" alt="FROST BYTE logo" width="200"/>
</p>

## Description

This repository contains a reproducible data-driven workflow, organized as a collection of Jupyter Notebooks. The workflow leverages snow water equivalent (SWE) measurements as predictors and streamflow observations as predictands, drawn from reliable datasets like CanSWE, NRCS, SNOTEL, HYDAT, and USGS. Gap filling for SWE datasets is done using quantile mapping from nearby stations and Principal Component Analysis is used to identify independent predictor components. These components are employed in a regression model to generate ensemble hindcasts of seasonal streamflow volumes. This workflow was applied by Arnal et al. to 75 river basins with a nival (i.e., snowmelt-driven) regime and with minimal regulation across Canada and the USA, for generating hindcasts from 1979 to 2021. This study presented a user-oriented hindcast evaluation, offering valuable insights for snow surveyors, forecasters, workflow developers, and decision-makers.

## Repository Structure

- 📂 `notebooks/`: Collection of Jupyter Notebooks detailing each step of the forecasting workflow.
- 📂 `scripts/`: Functions used in the data processing and analyses carried out in the Notebooks.
- 📂 `settings/`: Settings for running the forecasting workflow.
- 📂 `test_case_data/`: Sample data for running the forecasting workflow for two single river basins: the Bow River at Banff in Alberta, Canada, and the Crystal River Abv Avalanche Crk, Near Redstone in Colorado, USA.
- 📄 `requirements.txt`: Lists the Python packages required for reproducing the workflow.

## Getting Started
The steps below will help you to have a fully set-up environment to explore and interact with the Jupyter notebooks. If you're new to Jupyter notebooks, you might want to [check out some resources](https://jupyter.org/) on how to use them effectively.

1. **Clone the Repository**

  Begin by cloning the repository to your local machine. Use the command below in your terminal or command prompt:
   ```bash
   git clone https://github.com/lou-a/FROST-BYTE.git
   ```
   This command will create a copy of the repository in your current directory.
2. **Set Up Virtual Environment (Optional)**  

  It's a good practice to use a virtual environment for Python projects. This isolates your project's dependencies from other projects. To create and activate a virtual environment, run:
   ```bash
   python -m venv frostbyte
   source frostbyte/bin/activate  # For Windows, use `env\Scripts\activate`
   ```
   This step creates a new virtual environment named `frostbyte` and activates it.

3. **Install Dependencies**  

  With your virtual environment activated, install the required Python packages using:

   ```bash
   pip install -r requirements.txt
   ```

   This command reads the requirements.txt file and installs all the necessary packages to run the notebooks.

4. **Navigate to the Notebooks Directory**  
   ```bash
   cd notebooks/
   ```

5. **Start Jupyter Notebook**  

  Finally, start the Jupyter Notebook server:
   ```bash
   jupyter notebook
   ```
   This command will open a new tab in your default web browser with the Jupyter Notebook interface, where you can open and run the notebooks.

## Contribution

We welcome and appreciate contributions from the community! If you're interested in improving these notebooks or adding new features, here's how you can contribute:

1. Fork and Clone: Fork this repository to your account and clone it locally to make your changes.

2. Create a Feature Branch: For each new feature or significant change, create a separate branch in your repository. This helps in keeping track of different contributions and ensures that the main branch remains stable.

3. Discuss Major Changes: If you're considering a major change or addition, open an issue first. This way, we can have a discussion about the proposed changes, their impact, and how they fit into the project's roadmap.

4. Commit Your Changes: Make your changes in your feature branch and commit them with clear, descriptive commit messages. Your commit messages should succinctly explain the changes and their rationale.

5. Test Your Changes: Ensure that your changes do not break existing functionality. Adding tests for new features is highly encouraged.

6. Create a Pull Request: Once you're ready, submit a pull request to merge your changes into the main branch. Provide a clear description in the pull request about the changes and their purpose.

7. Code Review: Your pull request will be reviewed by the maintainers. Engage in the review process if there are comments or suggestions.

8. Merge: Once your pull request is approved, it will be merged into the main branch.

By contributing to this project, you agree to abide by its terms and conditions. Happy coding and forecasting!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

## Citation

If you use this workflow or any data therein, please consider citing our paper. The citation details can be found in the `CITATION.md` file.
To be added ...

## Contact

If you have any questions about using or running the workflow, or are willing to contribute, please contact louise.arnal[-at-]usask.ca
