# poetry install
# poetry run dvc pull model config

poetry run bentoml build --version 1.0
poetry run bentoml containerize hf-template-serve-img:1.0

docker run -it -d \
    -p 12345:3000 \
    --rm \
    --gpus all \
    --name hf_template_serve \
    hf-template-serve-img:1.0 serve