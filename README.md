# HF_LM_train_template
 허깅페이스 seq2seqLM, causalLM 구분 없이 바로 학습하는 템플릿

```
root
├── README.md
├── config
│   └── training_cfg.yaml  # 수정하기
└── src
    ├── dataloader.py  # 수정하기
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

추후에 bentoml을 사용한 서빙 부분을 추가할 예정입니다.
