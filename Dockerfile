FROM bhafsa/deep_nilmtk:cuda

COPY src /src
COPY dataset /dataset
COPY requirements.txt /requirements.txt

RUN pip install -r /requirements.txt

WORKDIR /src

RUN sudo mkdir /home/guestuser
RUN sudo chmod -R 777 /home/guestuser

CMD ["python", "experiment.py"]

