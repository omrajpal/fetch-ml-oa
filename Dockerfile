
FROM python:3.8
WORKDIR /app

COPY model.py /app/
COPY predict.py /app/
COPY data_daily.csv /app/
COPY templates/index.html /app/templates

RUN python -m pip install --upgrade pip
RUN python -m pip install Flask
RUN pip install --default-timeout=900 torch
RUN python -m pip install scikit-learn
RUN python -m pip install pandas

RUN python model.py

EXPOSE 5000
CMD ["python", "predict.py"]
