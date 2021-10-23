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

## Gather, assess, clean data

Data used for this story come from Inside Airbnb. Inside Airbnb is a mission-driven activist project to provide data that quantifies the impact of short-term rentals on housing and residential communities; and provides a platform to support advocacy for policies to protect cities from the effects of short-term rentals.
I had to fix nulls in many fields, which I did, but the most problematic issue to assess was the many listings that did not have a "description". I used the "description" field to understand how hosts perceive the importance of sharing accessibility information for their listing. I used the name of the listing in those cases as much information was there. Figure 1 shows the distribution of null values in the Dataset.

Data Cleaning
I dropped "neighbourhood_group", "last_review", "reviews_per_month", "license" fields due to the percentage of nulls and their relative low pertinence to this story. I renamed using "name_missing" all listings missing name information and "host_name_missing" all listings missing hostname information. I dropped outliers in the price column regarding listings exceeding 2000 euros per day. I dropped apartments available for more than 365 days a year and listings with availability less than one day, having less than five reviews in the last 12 months. I fixed listing availability with less than one day and more than five reviews in the previous 12 months by replacing it with mean availability. I dropped listings with no description, number of people accommodating and missing "reviews" in the last 12 months. I also dropped records without "description" and "reviews" in the previous 12 months. I fixed the rest of the descriptions by copying the name to the description field.

I used the Regular Expressions and Natural Language Processing Toolkit to process the listings description field and extract elevator information as a keyword. The procedure tells us how the host understands the importance of sharing elevator information in the description rather than the data itself. The result is a binary field saying if the listing involves elevator information not. I dropped the original description field once I completed the procedure.

I also created four binary variables out of the " room_type" field. I kept the initial column along with the new four new variables: "entire_home_apt", "hotel_room", "private_room", "shared_room". Here is an immediate visualization tool describing room type.
The final object to feed all graphic material is all listings grouped by neighbourhood. The entity "availability_365" determines the mean, minimum and maximum availability of a listing x days in the future. Note a listing may not be available because it has been booked by a guest or blocked by the host. The mean, minimum and maximum "minimum_nights" is the value of availability from the calendar (looking at 365 nights in the future). After normalizing the price by removing outliers, I have included mean, minimum and maximum fees. Lastly, I have summed up "entire_home_apt", "hotel_room", "private_room", "shared_room" per neighbourhood.

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
