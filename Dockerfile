FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.8.pytorch2

USER root

# Create directory and set ownership in one step
RUN mkdir -p /tmp/app && chown 1000:1000 /tmp/app
RUN mkdir -p /tmp/app/weights && chown 1000:1000 /tmp/app/weights

RUN wget -O /tmp/app/weights/yolov9c-seg.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt
RUN wget -O /tmp/app/weights/yolov9c.pt https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c.pt

USER 1000

RUN pip install --user \
    ultralytics \
    pyyaml

# docker build -t gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.21 -f Dockerfile .
# docker run -it gcr.io/viewo-g/piper/agent/runner/apps/yolov9:0.0.21 bash