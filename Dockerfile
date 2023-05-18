FROM python:3.6

ENV APP_HOME /ampliview
WORKDIR $APP_HOME
COPY . $APP_HOME

RUN apt-get update

RUN pip install -r requirements.txt