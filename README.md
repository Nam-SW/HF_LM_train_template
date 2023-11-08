# HF_LM_train_template
 허깅페이스 seq2seqLM, causalLM 구분 없이 바로 학습하는 템플릿

```
root
├── README.md
├── config
│   ├── serving_cfg.yaml  # 모델 학습 직후라면 필요 x
│   └── training_cfg.yaml  # 수정하기
└── src
    ├── data.py
    ├── dataloader.py  # 수정하기
    ├── runner.py  # 서빙할때 추가 전처리 필요시 수정
    ├── service.py  # 서빙할때 추가 기능 필요시 수정
    ├── train.py
    └── utils.py
```

dataloader.py의 load 함수 내부, `== user define ==` 로 표시된 영역 내부를 자신의 데이터에 맞게 수정하셔야 합니다.  
`training_cfg.yaml`는 입맛에 맛게 수정하시면 됩니다.  
wandb 사용이 가능합니다. 사용하지 않으려면 trainingargs에서 report_to, run_name을 지워주세요.  

`MODELCFG.model` 부분은 `PretrainedModel.from_pretrained` 항목을,  
`MODELCFG.tokenizer` 부분은 `PretrainedTokenizer.from_pretrained` 항목을,  
`TRAININGARGS` 부분은 `TrainingArguments`의 양식을 따릅니다.   
공식문서에서 참고해서 작성하면 됩니다.

## bentoml 서빙
```bash
# 실행
$ bash startup.sh
```
모델 학습 이후, bentoml을 통해 도커화 + 서빙까지 한번에 진행할 수 있습니다.  
`service.py`에 함수를 추가해 다양한 기능을 추가할 수 있으며, `runner.py`를 수정하여 모델의 세부 생성 과정을 작성할 수 있습니다.  
`startup.sh`에서 외부 포트를 변경할 수 있으며, 기본적으로 `12345`로 실행됩니다.

## request, response 명세
### input args
```bash
curl -X 'POST' \
    'http://127.0.0.1:12345/generate' \
    -d '{
    "input_text": [  # List[str]
        "입력 문장 1",
        ...
        "입력 문장 N"
    ],
    "generate_args": {  # optional, huggingface generate config 따름
        "max_length": 512,
        "do_sample": True,
        "penalty_alpha": 0.5,
        "top_k": 10,
    }

}'
```
참조: [huggingface generate config](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig)

### ouptut
```json
{
    "status": 0, # Literal[0, -1, -2]
    "result": [
        "생성 결과 1",
        ...
        "생성 결과 N"
    ]
}
```
status 0: 정상 처리  
status -1: input arg 에러  
status -2: 정상 처리  