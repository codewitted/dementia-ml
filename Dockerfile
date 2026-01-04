FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "ad-ensemble", "/bin/bash", "-c"]

COPY . .

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root"]
