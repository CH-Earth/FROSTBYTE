# FROSTBYTE Jupyter Notebooks

To explore FROSTBYTE, the best way is to navigate the Jupyter Notebooks in this section! The image below shows the methods implemented in each notebook. Following that is a brief text description, but open the notebooks themselves to see all steps your yourself.

For installation instructions, refer back to the [landing page](https://github.com/lou-a/FROSTBYTE). Test data has been included for a sample catchment in Canada and in the USA.


<p align="center">
<img src="NotebookMethods.png" alt="Methods figure" width="500"/>
</p>

#### 1. Regime Classification (`1_RegimeClassification.ipynb`)

- Circular statistics to identify river basins with nival (i.e., snowmelt-driven) regimes.
- Based on three peak event identification metrics.


#### 2. Streamflow Preprocessing (`2_StreamflowPreprocessing.ipynb`)

- Linear interpolation to fill in small gaps in daily discharge data.
- Calculation of seasonal streamflow volumes from daily discharge.


#### 3. SWE Preprocessing (`3_SWEPreprocessing.ipynb`)

- Utilizes SWE and precipitation station data for gap filling of SWE daily data.
- Precipitation is accumulated over the water year to be used as a proxy for SWE in the gap filling.
- Employs quantile mapping with neighboring station data for gap filling.


#### 4. Forecasting (`4_Forecasting.ipynb`)

- Uses principal component analysis (PCA) for processing SWE data into principal components, used as predictors.
- Employs ordinary least squares (OLS) regression for deterministic forecasting of streamflow volumes retrospectively.
- Generates ensemble hindcasts with adjustable ensemble size through ensemble dressing.


#### 5. Hindcast Verification (`5_HindcastVerification.ipynb`)

- Evaluates ensemble hindcasts using various deterministic and probabilistic verification metrics.
- Implements bootstrapping for uncertainty quantification.
