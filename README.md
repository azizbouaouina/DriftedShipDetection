# Drifted Ship Detection App

This is a desktop application built using Python that enables users to annotate datasets, create and train deep learning models, and utilize them for ship drift detection.

## Table of Contents
- [Features](#features)
- [Demo](#demo)
- [Installation](#installation)

## Features
- Annotating the dataset using the cursor
- Training deep learning models
- Testing the models on new data

## Demo
You can check out a demo of the app via these screenshots.

This is the home page of the app. 

![1](https://github.com/azizbouaouina/testtest/assets/104959387/244dce25-c12c-46ed-8d62-eaa87381bad4)

- When you click on "Annotation Dérive", you will be directed to the window below.

In this window, you can select CSV files for labeling. To simplify the process, use the cursor to specify the drift interval if it exists, instead of manually editing each line in the CSV file for every file.

![2](https://github.com/azizbouaouina/testtest/assets/104959387/4008b660-b68f-46e1-8ad8-e276ce734e6d)

![3](https://github.com/azizbouaouina/testtest/assets/104959387/a0ba0961-dce2-42be-86b3-135972fc9be6)

- When you click on "Entrainer et sauvegarder un modèle", you will be directed to the window below.

In this window, you can select the deep learning model you want to train and specify the hyperparameters to be used. If the results from the confusion matrix meet your satisfaction, you can save the model for future use on new data.

![4](https://github.com/azizbouaouina/testtest/assets/104959387/ff872011-80fe-4b76-9c7a-d12b030b5ad0)

- When you click on "Importer un modèle et le tester", you will be directed to the window below.

In this window, you can test saved models on untrained data and detect drift intervals for individual ships, multiple ships, or for the entire day of ship data.

![5](https://github.com/azizbouaouina/testtest/assets/104959387/9bc648c0-0574-42b9-9471-9153cc93177a)


## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/my-awesome-web-app.git

2. Run the 'Application Dérive.py' file, but please note that you won't be able to access the app's features as it requires additional files, including the dataset, which are not provided for privacy reasons.
