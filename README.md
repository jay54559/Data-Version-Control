# DVC Sample Project 1

This project is a sample utilizing DVC for building a pipeline for data classification using the iris dataset, which contains 3 types of iris plants. 

### Project Structure

The project structure is inspired from [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/). This allows for great code maintainability and better collaboration besides many other advantages.

### Package Management and Packaging

[Poetry](https://python-poetry.org/) is the tool that has been used for dependency management for this project. 

### DVC Pipeline

The whole training pipeline contains of the following stages:

1. data_load: This stage loads the appropriate raw data, and does some necessary processing with respect to column names.
2. featurize: This stage generates new features from the raw data.
3. data_split: This stage splits the data from the previous stage into train and test sets.
4. train: This stage runs the model training, and stores the model.
5. evaluate: This stage evaluates the model against test data and stores the results including a confusin matrix.

To run the pipeline, change to dvc-sample1/dvc-sample1 and run `dvc repro`