# dtsc330_26

## Homework 2:
Question 1: For the missing 'start_at' dates in the grants data, I would replace the missing date with the date of the closest row before it that is not missing its 'start_at' date. I would go about this by looping through each row, and if there is a NaN in the 'start_at' column, replace it with the previous non-NaN date. 

Question 2: To get "grantees", split each name up at the semicolon in PINAMEs using str.split(';') and have each grantee name in its own row. Use the .explode pandas command to do so.

Question 3: PubDate and DateCompleted in the Articles dataset are organized as tags within a tag. To parse the inner tags containing the year, month, day, hour, and minute values, there need to be nested for loops in the parse_article function. 

## Homework 3:
Performance using random_forest: 0.9240721824991488

The features I chose were heartrate, the absolute value of acc_x, acc_y, and acc_z, mean heartrate over a roughly 8 hour window, and the mean of each acceleration direction's absolute value over the same window. I chose these features because they provided a baseline average for each person in the experiment. I chose to take the absolute value of each acceleration datapoint because I thought it would improve performance.

## Homework 4:
Performance using XGBoost: 0.8934286687095676
Difference: roughly 3.1%

## Homework 5:
To limit the phonebook-to-phonebook matching problem, I would only compare entries where at least one variable (out of first name, last name, phone number, or address) matches, instead of doing an all-to-all comparison. It would probably be even more effective to only compare entries with a phone number match or last name match, as too many people change their address and have multiple ways to spell their first name. There still could be last name and phone number typos, but these errors are probably rare enough for it not to matter too much.

## Homework 9:
To provide training data for my model, I took 100 photos with my webcam while holding up a blue water bottle for the class 1 and 100 photos holding up a pink water bottle for class 2. Holding up either water bottle would give close to 100% confidence as output. It remarkably accurate. However, when I held up my arm without a water bottle, it would give about an 80-90% probablity of the pink water bottle depending on the angle I held my arm. This is likely due to my skin tone being much closer to the color of the pink bottle. Pulling up my sleeve shifted the probablity much closer to the blue water bottle likely because I was wearing a darker colored shirt. I think this classifer is able to perform well with limited training data for my model specifically because the bottle colors and shapes are so different. I think teachable machine in general performs as well as it does because the user provides labels for the classifier, and the actual classifier must be incredibily good with feature selection. With only a limited number of photos of water bottles. I'm guessing that behind the scenes, teachable machine is very good at pulling out color as a feature. 