# SignTalk AI backend for a Hugging Face Docker Space.
# Builds from the public GitHub repo, so the Space itself only holds this file and README.md.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH
WORKDIR /home/user/app

ARG REPO=https://github.com/Devyansh1103/SignTalk-AI.git
ARG BRANCH=main
RUN git clone --depth 1 --branch $BRANCH $REPO . && rm -rf .git

RUN pip install --no-cache-dir --user -r requirements-deploy.txt

# MediaPipe models are gitignored; fetch the same files used locally
RUN cd models/mediapipe && \
    for m in pose_landmarker/pose_landmarker_full hand_landmarker/hand_landmarker face_landmarker/face_landmarker; do \
      curl -fsSL -o "$(basename $m).task" "https://storage.googleapis.com/mediapipe-models/$m/float16/latest/$(basename $m).task"; \
    done

# Public server: never store visitors' landmarks; free Spaces have 2 vCPUs
RUN sed -i 's/save_segments: true/save_segments: false/; s/num_threads: 4 /num_threads: 2 /' configs/realtime.yaml

ENV CORS_ORIGIN_REGEX="https://.*\.vercel\.app"
EXPOSE 7860
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "7860"]
