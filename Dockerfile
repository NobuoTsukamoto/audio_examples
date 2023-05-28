FROM nvcr.io/nvidia/nemo:23.03

RUN apt-get update -y && \
    apt-get install -y sox libsndfile1 ffmpeg portaudio19-dev && apt-get clean && \
    pip3 install git+https://github.com/openai/whisper.git && \
    pip3 install sounddevice
