FROM conda/miniconda3:latest

COPY src/ .

RUN conda env create -f /src/make_cluster/environment.yml && conda init bash

WORKDIR /make_cluster
CMD ["bash", "run.sh"]
