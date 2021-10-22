## 1. Installations

In order to have this project up and running you need to have the following packages installed:
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Plotly Express](https://plotly.com/python/plotly-express/)
- [Chart Studio](https://plotly.com/python/getting-started-with-chart-studio/)
- [Regular Expressions](https://docs.python.org/3/library/re.html)
- [Natural Language Toolkit](https://www.nltk.org/)
- [Json](https://docs.python.org/3/library/json.html)
- [Time](https://docs.python.org/3/library/time.html)
- [Swifter](https://pypi.org/project/swifter/)

## 2. Project motivation

Airbnb data in Paris has been elaborated to answer the following questions:

1. Which are the most popular neighbours in Paris?
2. How do the prices range across neighbourhoods in Paris?
3. How likely is it for an apartment to have an elevator in different neighbourhoods?

There is an urgent need to raise awareness about mobility and accessibility and heavily involve technological development 
in the field to abolish any barrier and make our communities genuinely inclusive. 

A [medium blogpost](https://chpolyzo.medium.com/small-apartment-in-paris-378a4ae86073) has been written to support this project.


## 3. File descriptions

The [paris_listings](https://github.com/chpolyzo/DSND/blob/master/Blog_Post/paris/paris_listings.ipynb) jupyter notebook has to be run
to load clean and produce all visualizations.

Interactive visualizations are also stored in [my plotly profile](https://chart-studio.plotly.com/~chpolyzo)

All data is provided by [inside airbnb platform](http://insideairbnb.com/get-the-data.html).

You can find the compressed complete “listings” file on the platform under “listings.csv.gz”. When unzipped, 
you get a ~40 MB CSV file containing a “Description” field where you can find the host’s listing description.
I have merged this information with the “listings.csv” file, a ~2 MB CSV file. The result is a ~14 MB CSV file which, 
when cleaned to be ready for exploration, is ~1.4 MB file.

A previw of the original data can be found [here](http://insideairbnb.com/paris/).

## 4. How to interact with this project

Please install all necessary libraries described above and run the jupyter notebook 
[paris_listings](https://github.com/chpolyzo/DSND/blob/master/Blog_Post/paris/paris_listings.ipynb)

## 5. Licencing, Authors, Acknowledgments

The data behind the Inside Airbnb site is sourced from publicly available information from the Airbnb site. 
The data has been analyzed, cleansed and aggregated where appropriate to faciliate public discussion.
See more [disclaimers here](http://insideairbnb.com/about.html#disclaimers), 
and a [data dictionary here](https://docs.google.com/spreadsheets/d/1iWCNJcSutYqpULSQHlNyGInUvHg2BoUGoNRIGa6Szc4/edit#gid=982310896).
