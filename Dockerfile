FROM tensorflow/tensorflow:2.6.1-gpu-jupyter
ENV PYTHONUNBUFFERED=1
WORKDIR /usr/src/app
    
COPY requirements.txt ./
COPY . /usr/src/app

RUN pip3 install --upgrade pip 
RUN pip3 install -r requirements.txt 

EXPOSE 8888
