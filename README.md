# EMOTIONAL IMPACT OF MOVIES

This Thesis explores different approaches using deep learning techniques to predict emotions in videos.

Working with videos implies a huge amount of data including visual frames and acoustic samples. The first step of the project is basically to extract features to represent the videos in small sets of arrays. This procedure is done using pretrained models based on Convolutional Networks, the state of the art in visual recognition. Firstly, visual features are extracted using 3D convolutions and acoustic features are extracted using VGG19, a pretrained convolutional model for images fine-tuned to accept the audio inputs.

Later, these features are fed into Recurrent models that exploit the temporal information.

Emotions are measured in terms of valence and arousal, values between [-1, 1]. Additionally, the same techniques are also used to attempt to predict fear scenes. In consequence, this thesis deals with both regression and classification problems.

Several architectures and different parameters have been tested in order to achieve the best performance. Finally, the results will be published in the MediaEval 2017 Challenge and compared to the state-of-the-art solutions.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
```

### Installing

A step by step series of examples that tell you have to get a development env running

Say what the step will be

```
Give the example
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
Give an example
```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Keras](https://keras.io/) - The main framework used
* [Theano](http://deeplearning.net/software/theano/) - Backend Used
* [Python-telegram-bot](https://python-telegram-bot.readthedocs.io/en/latest/) - Used to send results

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

* **Oriol Bernal** - *Initial work* - [uribernal](https://github.com/uribernal)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.


## Acknowledgments

* Hat tip to anyone who's code was used
* Inspiration
* etc

