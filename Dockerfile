FROM tensorflow/tensorflow:2.4.1

RUN apt-get update -y

ADD requirements.txt /app/requirements.txt
RUN apt-get install -y libsndfile1-dev
RUN pip install -r /app/requirements.txt

ADD . /app
WORKDIR /app

EXPOSE 8000

CMD ["gunicorn","-b",":8000","app:faursound_app()"]

