FROM jenkins/jenkins:lts-jdk17
USER root
RUN apt-get update && \
    apt-get install -y python3.11 python3.11-pip python3.11-venv && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    rm -rf /var/lib/apt/lists/*
COPY plugins.txt /usr/share/jenkins/ref/plugins.txt
RUN jenkins-plugin-cli --plugin-file /usr/share/jenkins/ref/plugins.txt
USER jenkins