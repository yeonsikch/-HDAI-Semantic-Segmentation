본 Repo는 [HDAI](https://github.com/DatathonInfo/H.D.A.I.2021) 에서 진행한 Semantic Segmentation Competition에 대한 제출코드이다.
-------
## 테스트 방법
1. `python test.py --img_size 416 --device=0 --batch_size=12 --weight /path/to/model/weight --backbone efficientnet-b5 --domain a2c --input_path {{test data path}}`
2. `python test.py --img_size 416 --device=0 --batch_size=12 --weight /path/to/model/weight --backbone efficientnet-b5 --domain a4c --input_path {{test data path}}`
3. `{{test data path}}` : test를 진행할 이미지 폴더를 입력
-------
(예시) 아래와 같은 구성의 폴더라면 `{{test data path}}` 자리에는 `./echocardiography/test`를 입력한다.
```
--./echocardiography/test

--./echocardiography/test/A2C/

--./echocardiography/test/A4C/
```
4. `./detection-results` 폴더에 inference 결과 파일이 생성됨.

-------
[Notion](https://yeonsikc.notion.site/HDAI-Segmentation-40cbfebc919d4ca686a925fb4ec5c015)에 학습 전략에 대한 내용이 포함되어있습니다.
