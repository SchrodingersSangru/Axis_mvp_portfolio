# app/Dockerfile

FROM python:3.9-slim

EXPOSE 8501


RUN mkdir streamlit

COPY . /streamlit

WORKDIR /streamlit

RUN apt-get update && apt-get install -y \
    build-essential \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# RUN git clone https://github.com/.git .

RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "webapp_final.py", "--server.port=8501", "--server.address=0.0.0.0"]

# streamlit run portfolio_dashboard.py --server.port=8501 --server.address=0.0.0.0

# /home/sangram/Downloads/hello-main/Hello.py


## commands to run
# sudo docker build -t streamlit .
# sudo docker run -p 8501:8501 streamlit