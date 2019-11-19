FROM python:3

WORKDIR /usr/src/app/uploads

WORKDIR /usr/src/app

RUN pip install Werkzeug Flask flask-cors numpy Keras gevent pillow h5py tensorflow

WORKDIR /usr/src/app/models

COPY models .

WORKDIR /usr/src/app/frontend

COPY frontend .

WORKDIR /usr/src/app

COPY app.py .

CMD [ "python" , "app.py"]