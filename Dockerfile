FROM python:3.9-slim-buster
ENV DOCKER=True
COPY . .
RUN pip install -r requirements.txt
EXPOSE 80
CMD [ "python", "server.py"]
