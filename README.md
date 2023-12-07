## EDA manufacturing project

### Project description
This project is to optimise a manufacturing machine process for a large industrial company. We extract data on to the local machine from an RDS database. The data is then cleaned, analysed and visualised. Conclusions are then drawn on how the machine can be run with minimal failures.

### Installation instructions

Clone the github repostory by running the following command in a terminal.
```
git clone https://github.com/sgrayner/EDA-Manufacturing.git
```

### Data sources?

### Usage instructions

The notebook file, (Notebook.ipynb) is the file that will walk you through the exploratory data analysis process. Run the code blocks from the beginning to run the EDA process on the data.

The df_utils.py file contains the functions for extracting, cleaning, analysing and visualising the data.


### EDA functions

*class RDSDatabaseConnector*
- **read_creds** - Reads and returns the credentials of the RDS database where the data is stored.
- **init_db_engine** - Initiates an SQL database engine.
- **read_rds_table** - Reads and returns data from the RDS database.
- **save_data** - Saves the data to the local machine.
- **load_data** - Loads and returns the data as a Pandas dataframe.

*class DataTransform*
- **cleaning** - Cleans the data.
- **impute_nulls** - Imputes null values.
- **remove_skew** - Removes significant skew from the data.
- **drop_outliers** - Removes outlier values from the data.
- **remove_collinearity** - Removes variables that are collinear with others.

*class DataFrameInfo*
- **stats** - Returns central tendency and dispersion statistics.

*class DataAnalysis*
- **ranges_table** - Returns maximum and minimum values of explanatory variables.
- **failure_over_column_range** - Plots a histogram of a given type of machine failure superimposed with a histogram of a given explanatory variable.
- **failure_plots** - Plots bar charts detailing information on machine failure types as well as failures of each product quality type.


### File structure

### Results/findings

- Torque must be kept between 10 Nm and 42 Nm to prevent HDF, PWF and OSF
- Tool wear must be below 175 min to prevent TWF and OSF.
- Air temperature should be kept below 300 K to prevent HDF.
- Process temperature should be kept below 309.5 K to prevent HDF.
