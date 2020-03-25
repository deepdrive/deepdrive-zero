FROM pytorch/pytorch

#RUN mkdir /tmp/build/
#COPY . /tmp/build
#RUN find /tmp/build/

# Install OpenMPI per https://spinningup.openai.com/en/latest/user/installation.html#installing-openmpi
RUN apt-get update && apt-get install libopenmpi-dev -y

# Cache requirements
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install ssh for openmpi
RUN apt-get install ssh -y

COPY repos/deepdrive-zero deepdrive-zero
COPY repos/spinningup spinningup

RUN cd deepdrive-zero && pip install -e .
RUN cd spinningup && pip install -e .

#https://raw.githubusercontent.com/crizCraig/dotfiles/master/.inputrc