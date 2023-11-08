import bentoml
from hydra import compose, initialize
from bentoml.io import JSON

from src.runner import ClaimsGenerator
from src.data import BentomlDataType


with initialize(version_base="1.2", config_path="../config/"):
    cfg = compose(config_name="serving_cfg")

runner = bentoml.Runner(
    ClaimsGenerator,
    name="claim_generator",
    runnable_init_params=cfg.RUNNER,
)

svc = bentoml.Service(cfg.MODELS.service_name, runners=[runner])


@svc.api(input=JSON(pydantic_model=BentomlDataType), output=JSON())
def generate(input_args):
    input_args = dict(input_args)
    input_text = input_args.get("input_text")
    generate_args = input_args.get("generate_args")

    result_dict = dict()

    if input_text is None:
        result_dict = {"status": -1, "detail": "inputs must required."}

    else:
        status, result = runner.generate.run(input_text, generate_args)
        result_dict["status"] = status

        if status == -1:
            result_dict["detail"] = "`inputs` must list[str]."
        elif status == -2:
            result_dict["detail"] = result
        else:
            result_dict["result"] = result

    return result_dict
