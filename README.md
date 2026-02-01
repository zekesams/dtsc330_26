# dtsc330_26

## Homework 2:
Question 1: For the missing 'start_at' dates in the grants data, I would replace the missing date with the date of the closest row before it that is not missing its 'start_at' date. I would go about this by looping through each row, and if there is a NaN in the 'start_at' column, replace it with the previous non-NaN date. 

Question 2: To get "grantees", split each name up in PINAMEs and have each grantee name in its own row. Use the .explode pandas command to do so.

Question 3: PubDate in the Articles dataset is organized as tags within a tag. To parse the inner tags containing the year, month, day, hour, and minute values, there needs to be a nested for loop in the parse_article function. 
