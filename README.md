# Arrival delay time prediction of commercial flight
The aim of this repository is to create a model capable of predicting the arrival delay time of a commercial flight, given a set of parameters known at time of take-off. In order to achieve this, the publicly available data from commercial USA domestic flights [1] will be use. The main result of this work will be a Spark application, programmed to perform the following tasks:
* Load the input data, previously stored at a known location
* Select, process and transform the input variables, to prepare them for training the model
* Perform some basic analysis of each input variable
* Create a machine learning model that predicts the arrival delay time
* Validate the created model and provide some measure of its accuracy

## Run the application
The application is prepared to be run in a docker container. For this reason is require to install Docker and `docker-compose` before being able to run this project. After this requirements are met, the following command will create the master and worker containers:

```
docker-compose up
```

With the containers created, we need to attached a shell to the master container in order to be able to run the application. This can be achieve by running the following commands:
```
docker exec -it <container_id> sh
```

In the shell that will be open, type the following command and with that, the created application will run:

```bash
spark-submit --master spark://spark-master:7077 --class upm.bd.App app/target/scala-2.12/sparkapp_2.12-1.0.0.jar
```

If you want to see the states of the spark application, just go to a browser and enter to the link `localhost:8080`.

## References
[1] 2008, "Data Expo 2009: Airline on time data", https://doi.org/10.7910/DVN/HG7NV7, Harvard Dataverse, V1