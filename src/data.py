from typing import Dict, Optional

from pydantic import BaseModel


class BentomlDataType(BaseModel):
    input_text: list[str] = ["기본 문장"]
    generate_args: Optional[Dict] = None
