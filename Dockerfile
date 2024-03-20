FROM python:3.9.16-slim-buster

RUN apt update && apt install -y libpq-dev gcc ffmpeg libsm6 libxext6


WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN python -m pip install -r requirements.txt
RUN python -m pip install onnxruntime==1.12.0

COPY . .


CMD ["python",  "run.py"]
