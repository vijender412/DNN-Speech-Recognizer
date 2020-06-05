# DNN-Speech-Recognizer

We begin by investigating the LibriSpeech dataset that will be used to train and evaluate your models. Your algorithm will first convert any raw audio to feature representations that are commonly used for ASR. You will then move on to building neural networks that can map these audio features to transcribed text. After learning about the basic types of layers that are often used for deep learning-based approaches to ASR, you will engage in your own investigations by creating and testing your own state-of-the-art models.

In this notebook, you will build a deep neural network that functions as part of an end-to-end automatic speech recognition (ASR) pipeline! Your completed pipeline will accept raw audio as input and return a predicted transcription of the spoken language. The full pipeline is summarized in the figure below.

![alt text](images/pipeline.png)

* STEP 1 is a pre-processing step that converts raw audio to one of two feature representations that are commonly used for ASR.
* STEP 2 is an acoustic model which accepts audio features as input and returns a probability distribution over all potential transcriptions. After learning about the basic types of neural networks that are often used for acoustic modeling, you will engage in your own investigations, to design your own acoustic model!
* STEP 3 in the pipeline takes the output from the acoustic model and returns a predicted transcription.

# The Data
We begin by investigating the dataset that will be used to train and evaluate your pipeline. LibriSpeech is a large corpus of English-read speech, designed for training and evaluating models for ASR. The dataset contains 1000 hours of speech derived from audiobooks. We will work with a small subset in this project, since larger-scale data would take a long while to train. However, after completing this project, if you are interested in exploring further, you are encouraged to work with more of the data that is provided online.

In the code cells below, you will use the vis_train_features module to visualize a training example. The supplied argument index=0 tells the module to extract the first example in the training set. (You are welcome to change index=0 to point to a different training example, if you like, but please DO NOT amend any other code in the cell.) The returned variables are:

```
vis_text - transcribed text (label) for the training example.
vis_raw_audio - raw audio waveform for the training example.
vis_mfcc_feature - mel-frequency cepstral coefficients (MFCCs) for the training example.
vis_spectrogram_feature - spectrogram for the training example.
vis_audio_path - the file path to the training example.
```
