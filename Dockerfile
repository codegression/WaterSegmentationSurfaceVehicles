FROM python:3.9-slim-buster
ENV DOCKER=True
COPY . ./src
RUN pip install -r /src/requirements.txt
EXPOSE 80
CMD [ "python", "./src/server.py"]
