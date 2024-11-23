# fetch-oa
A simple flask application to host the receipt prediction machine learning model.

To run this website, ensure you have the proper dependincies installed. These are pandas, numpy, torch, sklearn, datetime, and calendar.

To create the model, run the following:

> python modelGeneration.py

Then, to create and host the website, run the following:

> python frontEnd.py

This will run a graph that displays the machine learning model's predictions in a Chart using Chart.js

If you'd like to access the Docker container for this application, it can be found at the following link:

https://hub.docker.com/repository/docker/jasmith55/fetch-oa/general
