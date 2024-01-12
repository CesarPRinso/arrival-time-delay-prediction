# Arrival delay time prediction of commercial flight
The aim of this repository is to create a model capable of predicting the arrival delay time of a commercial flight, given a set of parameters known at time of take-off. In order to achieve this, the publicly available data from commercial USA domestic flights [1] will be use. The main result of this work will be a Spark application, programmed to perform the following tasks:
* Load the input data, previously stored at a known location
* Select, process and transform the input variables, to prepare them for training the model
* Perform some basic analysis of each input variable
* Create a machine learning model that predicts the arrival delay time
* Validate the created model and provide some measure of its accuracy

 "Data Expo 2009: Airline on time data", https://doi.org/10.7910/DVN/HG7NV7, Harvard Dataverse, V1