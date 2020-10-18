# 뉴스 읽어주는 딥러닝(잠꼬대 하는 손석희 만들기) 

* 기존 carpedm20(김태훈님)의 github를 바탕으로 Window10에 맞게 코드를 수정했습니다.  
* 손석희 앵커 목소리로  모델 생성에는 성공했으나, 결과적으로 깨끗한 음성이 아니라 잠꼬대를 하는 것과 같은 우물우물한 목소리가 출력되었습니다. 결과가 좋지 않은 이유를 아래에 함께 서술하였습니다. 

## 필요한 프로그램 
- Python 3.6+
- FFmpeg
- Tensorflow 1.3

** Tensorflow 1.3를 pip install로 설치할 경우 avx미지원으로 인한 에러가 발생합니다. 코드는 돌아가지만 step별 속도가 눈에 띄게 느리기 때문에 bazel을 이용해서 직접 다운을 받는 것이 좋습니다. 또한 이러한 문제는 CPU를 사용할 때 발생하는 문제임으로 tensorflow-gpu를 사용하시면 이러한 에러가 나와도 무시하고 진행해도 좋다고 합니다.


## 모형 만들어보기 

### 0. 터미널(윈도우에서는 cmd)에서 코드를 받아옵니다.(코드가 다 저장되어있느 상태이기 때문에, 모두 터미널에서 실행만 시켜주면 됩니다.)
물론 깃을 한번도 설치한적이 없으시다면, 설치하셔야합니다. 윈도우의 경우는 따로 설치해야하며, 맥을 사용하는 경우는 아마 이미 설치되어있을 것입니다.

    git init 
    git pull https://github.com/carpedm20/multi-Speaker-tacotron-tensorflow

### 1. 환경 생성과 패키지 설치 
필요한 어마무시한 패키지들을 설치해줍니다. 친절하게도 이미 `requirements`에 버전까지 저장되어있습니다. 버전은 꼬일수 있으니 명시된대로 사용해주셔야합니다. 맘대로 업그레이드를 하시면 중간에 코드가 돌아가지 않을 가능성이 매우 큽니다. `nltk`는 자연어 처리를 위한 파이썬 패키지 입니다.  

    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt')" 


### 2-1. 데이터 다운로드

다음(daum)에 올라온 손석희 앵커브리핑 영상을 기반으로 tacotron모형에 대입해보고자 합니다. 손석희 앵커브리핑 데이터를 사용하는 이유는 대부분 손석희 앵커가 단독으로 말하기 때문입니다. (그러나 음악이 나오거나 영상을 틀면서 말하는 경우도 많기는 합니다...)

1. 손석희 앵커 음성파일 다운로드

다음의 코드(datasets/son/download.py코드를 실행해주는 코드)로 다음(daum)에 올라온 손석희 앵커브리핑 영상 wav파일과 해당 영상에서 추출된 오디오 mp3파일 그리고 대본(스크립트)을 얻을 수 있습니다.(영상은 3~5분정도입니다.)

       python -m datasets.son.download

2. 침묵구간마다 오디오 분할

음성파일을 문장단위로 잘라주고자 음성에서 오래 쉬는 부분을 기준으로 잘라주는데, 손석희 앵커가 문장을 빠른속도로 연이어서 말할때는 두문장이 
붙여서 저장되고 혹은 천천히 말하는 경우에는 단어별로 잘리기도 합니다. 더 큰 문제는 문장이 다 끝나지않고 애매한 구간에서 잘라준다는 것입니다. 
예) "오늘은 기분 좋은 날입~" 이렇게 끊김

       python -m audio.silence --audio_pattern "./datasets/son/audio/*.wav" --method=pydub

### 2-2. 데이터 프로세싱

위에서 얻은 분할된 오디오 파일에 음성인식을 해보고자 합니다.

0. 오디오 파일과 해당 오디오의 대본을 매칭시키려면 구글의 도움을 받아야합니다. ([Google Speech Recognition API](https://cloud.google.com/speech/) 사용 예정)

해당 api를 사용하려면 credential.json이 필요합니다. 받는방법은 다음 링크와 같습니다.[(클릭)](https://developers.google.com/identity/protocols/application-default-credentials).
credential.json을 받아온 후에는 다음과 같이 등록을 해줍니다. 

       export GOOGLE_APPLICATION_CREDENTIALS="나의 CREDENTIALS.json"

1. [Google Speech Recognition API](https://cloud.google.com/speech/)를 이용하여 위에서 얻어온 분할된 음성 파일에 음성인식을 해줍니다.
그런데 여기서 놀라운점은  "오늘은 기분 좋은 날입~"을 음성인식을 해주면 구글이 똑똑하게 "오늘은 기분 좋은 날입니다"로 들리지도 않은 "니다"를 붙여줍니다. 결과적으로는 해당 이슈로 음성 파일과 대본(스크립트)사이의 미스매치가 생기게 됩니다. 데이터가 정확하지 않은 것이죠. 그리고 음성인식을 잘못해주는 경우도 있습니다. 
 
       python -m recognition.google --audio_pattern "./datasets/son/audio/*.*.wav"

2. 태훈님께서는 아래의 코드를 통해서 인식된 텍스트와 진짜 텍스트를 비교하기 하고, 음성 파일과 텍스트 파일을 쌍으로 연결한 내용을 alignment.json 에 저장하는 과정을 거치셨습니다.
그러나 저는 이부분은 패스했습니다. 왜냐하면 일단 음성파일이 완벽하지 않고 위에서 얻은 음성인식 스크립트와 실제 손석희 앵커의 스크립트를 맞춰주는 과정인데 더 결과가 이상해졌습니다. 

       python -m recognition.alignment --recognition_path "./datasets/son/recognition.json" --score_threshold=0.5

5. 음원/음원을 변형한 스펙토그램/음성인식을 한부분을 numpy압축형식으로 변환하여 tacotron모형에 넣어줄 준비를 해줍니다. (숫자형태의 array로 저장)

       python -m datasets.generate_data ./datasets/son/alignment.json


결과적으로 아래와 같은 형식으로 데이터가 폴더에 정리됩니다.

    datasets
    ├── son
    │   ├── alignment.json
    │   └── audio
    │       ├── 1.mp3
    │       ├── 2.mp3
    │       ├── 3.mp3
    │       └── ...
    └── YOUR_DATASET
        ├── alignment.json
        └── audio
            ├── 1.mp3
            ├── 2.mp3
            ├── 3.mp3
            └── ...


이렇게 해서 아래 과정으로 학습을 시켜주면 결과가 좋지 못합니다. 왜냐하면 데이터 자체가 지금 양질의 정확한 데이터가 아니기때문입니다. 
따라서 5과정을 하기전에 일일이 듣고 문장이 완벽하게 끝나지 않은 음성파일은 제거하고 제대로 된 음성파일의 경우에는 음성인식의 결과를 고쳐주어야합니다. 띄어쓰기와 인식이 제대로 되지 않은 부분이 상당히 많습니다. 그러나 이 음성파일은 도합 30시간이 넘어서 저는 도저히 엄두가 나지않아 끝까지 처리하지 못했습니다.

### 3.1 모델 설명

![슬라이드11](https://user-images.githubusercontent.com/47768004/96369584-ee417a80-1195-11eb-9872-077346e6e500.jpg)
![슬라이드12](https://user-images.githubusercontent.com/47768004/96369589-f1d50180-1195-11eb-98cf-cd3527d8eb3b.jpg)
![슬라이드13](https://user-images.githubusercontent.com/47768004/96369591-f39ec500-1195-11eb-9e0c-6eb139d5cf7c.jpg)
![슬라이드14](https://user-images.githubusercontent.com/47768004/96369593-f699b580-1195-11eb-9acf-026020d99b0a.jpg)
![슬라이드16](https://user-images.githubusercontent.com/47768004/96369599-fc8f9680-1195-11eb-9199-734e3c8d0911.jpg)
![슬라이드17](https://user-images.githubusercontent.com/47768004/96369602-ff8a8700-1195-11eb-8981-91fc72ae38e8.jpg)
![슬라이드18](https://user-images.githubusercontent.com/47768004/96369608-0c0edf80-1196-11eb-98bd-38244848d9e0.jpg)
![슬라이드20](https://user-images.githubusercontent.com/47768004/96369614-0fa26680-1196-11eb-991e-23253ffb4226.jpg)
![슬라이드21](https://user-images.githubusercontent.com/47768004/96369615-116c2a00-1196-11eb-8a12-464649b7edd8.jpg)
![슬라이드22](https://user-images.githubusercontent.com/47768004/96369617-1630de00-1196-11eb-9507-fff528cfbc03.jpg)
![슬라이드24](https://user-images.githubusercontent.com/47768004/96369622-19c46500-1196-11eb-8ac5-1a2352f6e31a.jpg)
![슬라이드25](https://user-images.githubusercontent.com/47768004/96369625-1b8e2880-1196-11eb-817e-84cb28cefd67.jpg)
![슬라이드27](https://user-images.githubusercontent.com/47768004/96369627-1df08280-1196-11eb-8160-9132cfdf66b5.jpg)


### 3.2 모델 학습
다음의 코드로 모델을 학습시킬 수 있습니다.
모델의 중요한 하이퍼 파라미터는 `hparams.py`에 정의되어 있습니다.

    python train.py --data_path=datasets/son
    python train.py --data_path=datasets/son --initialize_path=PATH_TO_CHECKPOINT # 진행하던걸 이어서 진행시켜주고 싶을때

### 3.3 학습  결과

![슬라이드28](https://user-images.githubusercontent.com/47768004/96369661-5001e480-1196-11eb-8ec5-cc0a3789aff2.jpg)

이상적인 어텐션은 우상향으로 일직선으로 나타는 것인데, 학습을 진행해도 그러한 형식은 나타나지 않는 것으로 보였습니다.

![슬라이드29](https://user-images.githubusercontent.com/47768004/96369663-51331180-1196-11eb-8f8a-8818724df4db.jpg)
![슬라이드30](https://user-images.githubusercontent.com/47768004/96369666-52fcd500-1196-11eb-9003-81873cc15f52.jpg)
![슬라이드33](https://user-images.githubusercontent.com/47768004/96369671-57c18900-1196-11eb-9b26-9f6289072c3d.jpg)



### 모범 결과

태훈님의 결과를 첨부드리면 다음과 같습니다. 학습이 진행될수록 플롯이 우상향으로 일직선으로 이상적인 어텐션을 보입니다.

Training attention on single speaker model): 

![model](./assets/attention_single_speaker.gif)

Training attention on multi speaker model:

![model](./assets/attention_multi_speaker.gif)


## 레퍼런스

- [Keith Ito](https://github.com/keithito)'s [tacotron](https://github.com/keithito/tacotron)
- [DEVIEW 2017 presentation](https://www.slideshare.net/carpedm20/deview-2017-80824162)


## 원작자

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
