# movie-revenue-model
An ML powered model used to predict movies revenue and classify their success

## Project Overview
This project is a machine learning model that predicts the movie revenue and classifies the movie success. The model is trained using different regression and classification algorithms.

## Dataset
The dataset is provided by the FCIS ML Team as a part of the FCIS ML course project. The dataset contains 463 movies and their features.

The dataset used for this project can be found in the repository [here](Model/datasets/1/train/) for regression and [here](Model/datasets/2/train/) for classification.

## Features & Preprocessing

- Handling multiple datasets:
  - Used the movie name as the primary key between the three tables to merge them into one dataframe.
  - Transformed the voice-actors database into one column to indicate whether its an animated movie or not.

- Director:
  - Used [`TMDB API`](https://developers.themoviedb.org/3/getting-started/introduction) to **fill missing directors**.
  - Replaced the director with his or her TMDB popularity score.

- Release Date:
  - Extracted day, month, season columns from it.
  - Transformed it to a scalar by converting the days to a fraction of a year.

- Movie revenue:
  - Used [`cpi`](https://palewi.re/docs/cpi/) library to adjust the movie revenue to inflation using the release date. 


## Libraries Used

- pandas
- numpy
- cpi
- skcikit-learn
- matplotlib
- seaborn
- pickle

## Project Reports
The project reports can be found [here](Reports/). The report contains how the project was developed, what were our approaches,
what features were used, how did we preprocess the data, the algorithms used, the results, and the conclusion.

## Results
### Regression Metrics

| Model | Train MSE | Val MSE | Test MSE | Train R2 | Val R2 | Test R2 | 
| --- | --- | --- | --- | --- | --- | --- |
| XGBoost 			        | --- | --- | --- | --- | --- | --- |
| GradientBoosting 	    | --- | --- | --- | --- | --- | --- |
| PolynomialRegression 	| --- | --- | --- | --- | --- | --- |
| ElasticNet 		        | --- | --- | --- | --- | --- | --- |
| Linear Regression 	  | --- | --- | --- | --- | --- | --- |
| CatBoost 			        | --- | --- | --- | --- | --- | --- |


### Classification Metrics

| Model | Train Accuracy | Validation Accuracy | Test Accuracy |
| ---                   | --- | --- | --- |
| SVC                   | --- | --- | --- |
| RandomForest          | --- | --- | --- |
| LogisticRegression    | --- | --- | --- |
| CatBoost              | --- | --- | --- |

## License
[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
