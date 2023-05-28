# ampliview

Ampliview is a tool to help you write better reviews for Yelp. It offers two major functions: it will analyze and tag the sentiment of each sentence in your review ([read more](https://ampliview.herokuapp.com/sentiment)), and it will provide an overall score for your review on how likely it is to be marked as useful by reviewers on Yelp ([read more](https://ampliview.herokuapp.com/useful-score)). Check it out live: [https://ampliview.herokuapp.com/](https://ampliview.herokuapp.com/).

## Running

Create Container:

```
make build
```

Start local server:

```
make server
```


## Pyenv

```
pyenv install 3.10.11
pyenv virtualenv 3.10.11 ampliview
pyenv activate ampliview
pip install -r requirements.txt
```
