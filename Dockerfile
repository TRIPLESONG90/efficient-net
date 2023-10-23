# basedon pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /src

#copy src folder
COPY src /src
COPY requirements.txt /requirements.txt

#install requirements
RUN pip install -r /requirements.txt

#opencv python dependencies
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get install -y libglib2.0-0

