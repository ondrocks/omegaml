# adopted from https://github.com/jupyterhub/the-littlest-jupyterhub/blob/master/integration-tests/Dockerfile
FROM jupyterhub/jupyterhub:latest
ARG  pypi
ENV  pypi=$pypi
COPY . /app
COPY ./packages /var/packages
RUN pip install --upgrade pip -q
RUN pip install --ignore-installed six -f /var/packages -i ${pypi:-https://pypi.org/simple/} --extra-index-url https://pypi.org/simple/ --pre omegaml[all]
RUN pip install notebook jupyterlab numpy<1.17
RUN useradd -ms/bin/bash admin && \
    echo "admin:test" | chpasswd admin && \
    touch /app/config.yml
RUN /app/setupnb.sh
