The primary goal of my project is to develop a machine learning model capable of identifying individuals at risk of cerebral stroke, and to compare the effectiveness of two models in making this prediction. In essence, the project aims to facilitate early interventions and preventive measures by healthcare professionals.
For this analysis, I leveraged on Python as my programming language, and I used both linear regression and Random Forest algorithms for my predictive models.
My initial steps involved thorough examination of the data's quality and consistency. Subsequently, I performed data cleaning procedures to rectify any inconsistencies present in the data. Following the data cleaning, I delved into data exploration to glean insights from the dataset. Here are some notable findings from this exploration:

1. Females exhibit a higher likelihood of developing cerebral stroke compared to men.
2. Individuals employed in private firms are prone to cerebral stroke, with self-employed individuals following closely behind.
3. The risk of cerebral stroke appears to increase steadily from the age of 28 onwards.

To prepare the dataset for model development, I converted all categorical values into numerical representations using the 'LabelEncoder' technique. Furthermore, I normalized the dataset to ensure that all features contribute equally to the model. Afterwards, I developed my models.

Upon completion of model development, I evaluated the performances of both linear regression and Random Forest regression models. Interestingly, linear regression outperformed Random Forest regression in this predictive task, giving a higher accuracy and effectiveness in cerebral stroke prediction.
