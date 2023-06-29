# Products-Top-Flop-Prediction
A fashion e-commerce company is planning its collections for the upcoming year. Therefore the company put together many potential products as candidates and now would like to estimate which products would be successful (top) or not (flop). To do so, you are provided with data on the past years’ top and flop products. This will allow us to create a small machine-learning application.

## Data Overview (shared in separate files)
We have two data sets:
▪ Historic data: Products of the past two years and their attributes (including a label that categories
the item stop or flop); file: historic.csv (8000 products)
▪ Prediction data: Potential products of the upcoming year and their attributes (but no label about the
success); file: prediction_input.csv (2000 product candidates)

## Columns:
▪ item_no: Internal identifier for a past product or a product candidate for the
future.
▪ category: Category of the product.
▪ main_promotion: Main promotion that would be/was used to promote the product.
▪ color: The main color of the product.
▪ stars: Stars of reviews from a comparable product of a competitor (from 0= very negative reviews to
5 = very positive reviews).
▪ success_indicator: Indicatorwhether a product wassuccessful(top) or not(flop) in the past. Only given
for the historic data
