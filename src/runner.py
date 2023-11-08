from typing import Tuple, Dict, Optional

import bentoml

from src.utils import load_pipeline, set_seed


class BentomlCustomLunner(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("nvidia.com/gpu",)
    SUPPORTS_CPU_MULTI_THREADING = False

    def __init__(
        self,
        cfg: Dict,
        default_generate_args: Dict,
        batch_size: int = 4,
        default_seed: int = 42,
    ):
        self.pipeline = load_pipeline(cfg)

        self.default_generate_args = default_generate_args
        self.batch_size = batch_size
        self.default_seed = default_seed

    @bentoml.Runnable.method(batchable=False)
    def generate(self, inputs, generate_args) -> Tuple[int, Optional[Dict]]:
        status = 0
        result = None
        if generate_args is None:
            generate_args = self.default_generate_args

        if not isinstance(inputs, (list, tuple)) or sum([not isinstance(s, str) for s in inputs]):
            return -1, result

        try:
            set_seed(self.default_seed)
            result = self.pipeline(inputs, batch_size=self.batch_size, **generate_args)
            result = [
                (g[0] if isinstance(g, list) else g)["generated_text"].replace("|", "\n")
                for g in result
            ]

        except Exception as e:
            status = -2
            result = str(e)

        return status, result
