FROM adamliter/psiturk:latest

RUN pip3 install pandas scipy matplotlib seaborn spacy
RUN python3 -m spacy download en

COPY materials /materials
COPY psiturk /psiturk
RUN rm /psiturk/.psiturkconfig

ARG SEQUENCE_LENGTH=3
WORKDIR /psiturk
RUN python3 materials.py -i $SEQUENCE_LENGTH
