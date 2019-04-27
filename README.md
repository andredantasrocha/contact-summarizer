## Install the dependencies
```
pip install -r requirements.txt
```

## Text Summarization

### Setup

```
python
>>> import nltk
>>> nltk.download('stopwords')
>>> nltk.download('punkt')
```

Download Spacy language model
```
python -m spacy download en
```

### API

Start the server
```
python api.py
```

Post the text to summarize
```
curl -X POST -H "Content-Type: text/plain; charset=UTF-8" --data-binary @example.txt http://localhost:5000/contact-summarizer/summarize
```

## Speech Recognition

### Simple
```
python speech_recognition.py
```

### DeepSpeech
This only works with very short audio files (4-5 seconds). A longer audio should be broken into multiple files. See the solution [here](https://discourse.mozilla.org/t/longer-audio-files-with-deep-speech/22784)

Download DeepSpeech model
```
cd <project_dir>

wget https://github.com/mozilla/DeepSpeech/releases/download/v0.4.1/deepspeech-0.4.1-models.tar.gz
tar xvfz deepspeech-0.4.1-models.tar.gz
```

Convert the audio to text
```
deepspeech --model models/output_graph.pbmm --alphabet models/alphabet.txt --lm models/lm.binary --trie models/trie --audio my_audio_file.wav
```
