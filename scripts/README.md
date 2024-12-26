# Analysis Scripts Modules

## Overview

The `eda_task_1` module provides a comprehensive set of functions designed to facilitate exploratory data analysis (EDA) and visualization of insurance data. These functions help in loading, summarizing, cleaning and handling, and visualizing the data to extract valuable insights.

## Functions

### 1. `load_data(file_path, delimiter='|')`
#### Description:
Loads data from a text file with a specified delimiter.

#### Parameters:
- `file_path` (str): The path to the text file.
- `delimiter` (str): The delimiter used in the text file (default is '|').

#### Returns:
- `data` (DataFrame): A pandas DataFrame containing the loaded data.

#### Usage:
data = load_data('MachineLearningRating_v3.txt', delimiter='|')



### 2. check_missing_values(data)
#### Description:
Checks for missing values in the dataset.

#### Parameters:
data (DataFrame): The DataFrame to be checked.

#### Returns:
Prints the count of missing values for each column.

#### Usage:
check_missing_values(data)

### 3. handle_missing_values(data)
#### Description:
Handles missing values by imputing or removing them.

#### Parameters:
data (DataFrame): The DataFrame with missing values.

#### Returns:
data (DataFrame): The DataFrame with missing values handled.

#### Usage:
data = handle_missing_values(data)


### 4. plot_histograms(data, columns)
#### Description:
Plots histograms for numerical columns.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

columns (list): A list of column names for which histograms are to be plotted.

#### Usage
plot_histograms(data, ['TotalPremium', 'TotalClaims'])

### 5. plot_correlation_matrix(data, columns)
#### Description:
Plots a correlation matrix for selected columns.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

columns (list): A list of column names to be included in the correlation matrix.

#### Usage:
plot_correlation_matrix(data, ['TotalPremium', 'TotalClaims', 'SumInsured'])

### 6. plot_boxplots(data, columns)
#### Description:
Plots box plots for numerical columns to detect outliers.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

columns (list): A list of column names for which box plots are to be plotted.

#### Usage:
plot_boxplots(data, ['TotalPremium', 'TotalClaims'])

### 7. plot_scatter(data, x, y, hue=None)
#### Description:
Plots a scatter plot for two numerical columns, with an optional hue.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

x (str): The column name for the x-axis.

y (str): The column name for the y-axis.

hue (str): The column name for color coding (optional).

#### Usage
plot_scatter(data, x='TotalPremium', y='TotalClaims')

### 8. plot_boxplot(data, x, y)
#### Description:
Plots a box plot for a categorical column vs. a numerical column, adjusting the figure size and rotating x-axis labels vertically for better readability.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

x (str): The column name for the x-axis (categorical).

y (str): The column name for the y-axis (numerical).

#### Usage:

plot_boxplot(data, x='Province', y='TotalClaims')

### 9. plot_bar(data, x, y, hue)
#### Description:
Plots a bar plot for a categorical column vs. a numerical column with hue, adjusting the figure size and rotating x-axis labels vertically for better readability.

#### Parameters:
data (DataFrame): The DataFrame containing the data.

x (str): The column name for the x-axis (categorical).

y (str): The column name for the y-axis (numerical).

hue (str): The column name for color coding (categorical).

#### Usage
plot_bar(data, x='VehicleType', y='TotalPremium', hue='Gender')