# Movie Lens Recommendation Engine

Learning and experimenting how to use Matrix Factorization (SVD) on a dataset in order to to make movie recommendations for a particular user. An additional feature is doing the inverse; finding which users are most likely to watch a particular movie, with the end goal of using this algorithm in an ad-manager application. Experimentation was done using both the Surprise library and the SciPy Linear Algebra library for SVD.

## Getting Started

Download or clone the repository, and simply run using python to see top-10 recommendations for both users and movies. Python 2.7 was used
```
python recommender.py
```
This will run the code for the algorithm using the SciPy library.
```
python surprise_recommender.py
```
This will run the code for the algorithm using the Surprise library

### Prerequisites

The following libraries must be installed:
* Pandas/Numpy
* Surprise
* SciPy
I would recommend downloading Conda for all these libraries

### Contributing

Just submit an issue you think is applicable. I am still learning, so criticism is welcome.

## Acknowledgments

* [Nick Becker](https://beckernick.github.io/matrix-factorization-recommender/) for the code using SciPy 
* [Surprise FAQ](http://surprise.readthedocs.io/en/stable/FAQ.html#how-to-get-the-top-n-recommendations-for-each-user) for getting Top-N Recommendations for each user

