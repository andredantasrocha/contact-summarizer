import speech_recognition as sr

if __name__ == "__main__":
    r = sr.Recognizer()
    harvard = sr.AudioFile('data/audio_files/harvard.wav')
    with harvard as source:
        audio = r.record(source)
        text = r.recognize_sphinx(audio)
        print(text)
