FROM python:latest

RUN pip install --upgrade pip

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "app.py"]