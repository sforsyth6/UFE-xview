FROM nvcr.io/nvidia/pytorch:19.09-py3

RUN /bin/bash -c 'pip install Pillow'
RUN /bin/bash -c 'conda install tsnecuda cuda101 -c cannylab'
RUN /bin/bash -c 'conda install faiss-gpu cudatoolkit=10.0 -c pytorch'
