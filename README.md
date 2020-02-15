# 뉴스 읽어주는 딥러닝(잠꼬대 하는 손석희 만들기) 

* 기존 carpedm20(김태훈님)의 github를 바탕으로 Window10에 맞게 코드를 수정했습니다.  
* 손석희 앵커 목소리로  모델 생성에는 성공했으나, 결과적으로 깨끗한 음성이 아니라 잠꼬대를 하는 것과 같은 우물우물한 목소리가 출력되었습니다. 결과가 좋지 않은 이유를 아래에 함께 서술하였습니다. 

## 필요한 프로그램 
- Python 3.6+
- FFmpeg
- [Tensorflow 1.3](그냥 pip install로 설치할 경우 avx미지원으로 인한 에러가 발생합니다. 코드는 돌아가지만 step별 속도가 눈에 띄게 느리기 때문에 bazel을 이용해서 직접 다운을 받는 것이 좋습니다. 또한 이러한 문제는 CPU를 사용할 때 발생하는 문제임으로 tensorflow-gpu를 사용하시면 이러한 에러가 나와도 무시하고 진행해도 좋다고 합니다.)


## 모형 만들어보기 

### 1. 환경 생성과 패키지 설치 
    
    pip install -r requirements.txt
    python -c "import nltk; nltk.download('punkt')" 


### 2-1. Generate custom datasets

아래와 같은 형식으로 데이터가 폴더에 정리됩니다.

The `datasets` directory should look like:

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

and `YOUR_DATASET/alignment.json` should look like:

    {
        "./datasets/YOUR_DATASET/audio/001.mp3": "My name is Taehoon Kim.",
        "./datasets/YOUR_DATASET/audio/002.mp3": "The buses aren't the problem.",
        "./datasets/YOUR_DATASET/audio/003.mp3": "They have discovered a new particle.",
    }

After you prepare as described, you should genearte preprocessed data with:

    python -m datasets.generate_data ./datasets/YOUR_DATASET/alignment.json


### 2-2. Generate Korean datasets

Follow below commands. (explain with `son` dataset)

 음성파일을 다운로드하여 문장단위로 나누고 음성인식을 해주어 진행할 것입니다.

0. To automate an alignment between sounds and texts, prepare `GOOGLE_APPLICATION_CREDENTIALS` to use [Google Speech Recognition API](https://cloud.google.com/speech/). To get credentials, read [this](https://developers.google.com/identity/protocols/application-default-credentials).

       export GOOGLE_APPLICATION_CREDENTIALS="YOUR-GOOGLE.CREDENTIALS.json"

1. Download speech(or video) and text. 

datasets/son/download.py코드를 실행해주는 코드로 손석희 앵커의 브리핑 영상과 함께 해당 영상에 따른 음성파일과 스크립트가 동시에 저장됩니다. 

       python -m datasets.son.download

2. Segment all audios on silence.

음성파일을 문장단위로 잘라주고자 음성에서 오래 쉬는 부분을 기준으로 잘라주는데, 손석희 앵커가 문장을 빠른속도로 연이어서 말할때는 두문장이 
붙여서 저장되고 혹은 천천히 말하는 경우에는 단어별로 잘리기도 합니다. 더 큰문제는 문장이 다 끝나지않고 애매한 구간에서 잘라준다는 것입니다. 
예) "오늘은 기분 좋은 날입~" 이렇게 끊김

       python -m audio.silence --audio_pattern "./datasets/son/audio/*.wav" --method=pydub

3. By using [Google Speech Recognition API](https://cloud.google.com/speech/), we predict sentences for all segmented audios.

그리고 위에서 얻어온 음성파일을 음성인식 해줍니다. 그런데 여기서 놀라운점은  "오늘은 기분 좋은 날입~"을 음성인식을 해주면 구글이 똑똑하게 
 "오늘은 기분 좋은 날입니다"로 들리지도 않은 "니다"를 붙여줍니다.
 
       python -m recognition.google --audio_pattern "./datasets/son/audio/*.*.wav"

4. By comparing original text and recognised text, save `audio<->text` pair information into `./datasets/son/alignment.json`.

저는 이부분은 패스했습니다. 왜냐하면 일단 음성파일이 완벽하지 않고 위에서 얻은 음성인식 스크립트와 실제 손석희 앵커의 스크립트를 맞춰주는 과정인데 더 결과가 이상해졌습니다. 

       python -m recognition.alignment --recognition_path "./datasets/son/recognition.json" --score_threshold=0.5

5. Finally, generated numpy files which will be used in training.

음원/음원을 변형한 스펙토그램/음성인식을 한부분을 numpy압축형식으로 변환하여 nnet에 넣어줄 준비를 해줍니다. 

       python -m datasets.generate_data ./datasets/son/alignment.json

Because the automatic generation is extremely naive, the dataset is noisy. However, if you have enough datasets (20+ hours with random initialization or 5+ hours with pretrained model initialization), you can expect an acceptable quality of audio synthesis.

이렇게 해서 아래 과정으로 학습을 시켜주면 결과가 좋지 못합니다. 왜냐하면 데이터 자체가 지금 양질의 정확한 데이터가 아니기때문입니다. 
따라서 5과정을 하기전에 일일이 듣고 문장이 완벽하게 끝나지 않은 음성파일은 제거하고 제대로 된 음성파일의 경우에는 음성인식의 결과를 고쳐주어야합니다. 띄어쓰기와 인식이 제대로 되지 않은 부분이 상당히 많습니다. 그러나 이 음성파일은 도합 30시간이 넘어서 저는 도저히 엄두가 나지않아 끝까지 처리하지 못했습니다.


### 3. Train a model

The important hyperparameters for a models are defined in `hparams.py`.

(**Change `cleaners` in `hparams.py` from `korean_cleaners` to `english_cleaners` to train with English dataset**)

To train a single-speaker model:

    python train.py --data_path=datasets/son
    python train.py --data_path=datasets/son --initialize_path=PATH_TO_CHECKPOINT

To train a multi-speaker model:

    # after change `model_type` in `hparams.py` to `deepvoice` or `simple`
    python train.py --data_path=datasets/son1,datasets/son2

To restart a training from previous experiments such as `logs/son-20171015`:

    python train.py --data_path=datasets/son --load_path logs/son-20171015

If you don't have good and enough (10+ hours) dataset, it would be better to use `--initialize_path` to use a well-trained model as initial parameters.



## Results

Training attention on single speaker model:

![model](./assets/attention_single_speaker.gif)

Training attention on multi speaker model:

![model](./assets/attention_multi_speaker.gif)


## Disclaimer

This is not an official [DEVSISTERS](http://devsisters.com/) product. This project is not responsible for misuse or for any damage that you may cause. You agree that you use this software at your own risk.


## References

- [Keith Ito](https://github.com/keithito)'s [tacotron](https://github.com/keithito/tacotron)
- [DEVIEW 2017 presentation](https://www.slideshare.net/carpedm20/deview-2017-80824162)


## Author

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
