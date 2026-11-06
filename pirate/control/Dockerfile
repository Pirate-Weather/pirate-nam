# The daily trigger. Small on purpose: all the heavy work happens on a throwaway GPU machine.
# Build from pirate/: docker build -f control/Dockerfile -t pirate-nam-control .
FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends openssh-client >/dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install -q --no-cache-dir "boto3==1.43.123" "ovh==1.2.0"
# ssh refuses to run for a user id with no passwd entry; 1000 also owns the mounted key.
RUN useradd -u 1000 -m pirate
COPY control /app/control
COPY prod /app/prod
COPY publish /app/publish
COPY cycle /app/cycle
COPY inputs /app/inputs
USER pirate
WORKDIR /app
ENV PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "/app/control/daily.py"]
