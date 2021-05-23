FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN mkdir /root/.jupyter

COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py

RUN conda install -y -n base -c conda-forge jupyterlab widgetsnbextension ipywidgets \
  && conda clean -ya

RUN pip install \
  pytorch_lightning \
  ray[tune] \
  Flask

WORKDIR /workspace

CMD ["jupyter", "lab", "--allow-root"]
