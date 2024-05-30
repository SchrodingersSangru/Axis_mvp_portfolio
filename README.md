To deploy it on the cloud, 
1. connect your device with ubuntu server with ssh login.

2. clear the repo, docker images and every other stuff.

3. Run the following commands to, build a docker and run it. 

sudo docker build -t streamlit .
sudo docker run -p 8501:8501 streamlit

where streamlit is a image name. 
