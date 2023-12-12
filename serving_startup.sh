version="1.0"  # 파일 내에서 직접 수정하는게 나으려나
port=${1:-51001}

poetry install
# poetry run dvc pull model config

poetry run bentoml build --version ${version}
poetry run bentoml containerize hf-template-serve-img:${version}

docker run -it -d \
    -p $port:3000 \
    --rm \
    --gpus all \
    --name hf_template_serve \
    hf-template-serve-img:${version} serve

