FROM python:3.8-slim-buster

COPY ./requirements.txt /
RUN pip install -r requirements.txt

COPY ./app /python_server

WORKDIR /python_server
EXPOSE 5000
CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]