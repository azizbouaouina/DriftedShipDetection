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

![1](https://github.com/azizbouaouina/DriftedShipDetection/assets/104959387/2483f955-eb1f-4ff2-a868-f54bb8810142)

- When you click on "Annotation Dérive", you will be directed to the window below.

In this window, you can select CSV files for labeling. To simplify the process, use the cursor to specify the drift interval if it exists, instead of manually editing each line in the CSV file for every file.

![2](https://github.com/azizbouaouina/DriftedShipDetection/assets/104959387/0cd0bd75-be34-4653-97e1-755a1f7d55a7)

![3](https://github.com/azizbouaouina/DriftedShipDetection/assets/104959387/3eb05e9e-419c-480d-8ceb-f52b7d4bb46f)

- When you click on "Entrainer et sauvegarder un modèle", you will be directed to the window below.

In this window, you can select the deep learning model you want to train and specify the hyperparameters to be used. If the results from the confusion matrix meet your satisfaction, you can save the model for future use on new data.

![4](https://github.com/azizbouaouina/DriftedShipDetection/assets/104959387/6943e4d6-cd89-4733-ae5b-2f26997c2f61)

- When you click on "Importer un modèle et le tester", you will be directed to the window below.

In this window, you can test saved models on untrained data and detect drift intervals for individual ships, multiple ships, or for the entire day of ship data.

![5](https://github.com/azizbouaouina/DriftedShipDetection/assets/104959387/1ea16362-21a7-4d3c-806b-7f2982a3f5b4)


## Installation
To run this project locally, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/my-awesome-web-app.git

2. Run the 'Application Dérive.py' file, but please note that you won't be able to access the app's features as it requires additional files, including the dataset, which are not provided for privacy reasons.
