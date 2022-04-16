FROM python:3.10.1-slim-buster

COPY ./requirements.txt /
RUN pip install -r requirements.txt

COPY ./app /python_server

WORKDIR /python_server
EXPOSE 8001
CMD ["python", "main.py"]
