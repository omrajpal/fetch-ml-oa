# Using python image
FROM python:3.8
WORKDIR /app
#Copying over files
COPY modelGeneration.py /app/
COPY frontEnd.py /app/
COPY data_daily.csv /app/
COPY templates/index.html /app/templates

#Installling dependencies
RUN python -m pip install --upgrade pip
RUN python -m pip install Flask
RUN pip install --default-timeout=900 torch
RUN python -m pip install scikit-learn
RUN python -m pip install pandas

#Running model to save for frontEnd.py
RUN python modelGeneration.py

EXPOSE 5000
CMD ["python", "frontEnd.py"]
