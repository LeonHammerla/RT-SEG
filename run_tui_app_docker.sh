docker build -f docker/Dockerfile -t mytui:gpu .
docker run -it --rm --gpus all mytui:gpu