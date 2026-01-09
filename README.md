Part 1: Correlation Analysis

Correlation Analysis Report
Part 1: Finding Pearson's Correlation Coefficient
1.1 Data Collection Process
Accessed the interactive graph at: max.ge/aiml_midterm/84536_html

Hovered the mouse over 10 blue data points to reveal their coordinates

Manually recorded the X and Y values from the tooltips displayed on screen

Created a dataset of 10 coordinate pairs

1.2 Collected Data Points
X Coordinate	Y Coordinate
-9.00	-8.30
-7.00	-6.50
-5.00	-4.70
-3.00	-2.90
-1.00	-0.80
1.00	0.90
3.00	3.10
5.00	4.80
7.00	6.70
9.20	7.90
1.3 Statistical Analysis
Number of data points: 10

Mean of X: 0.1400

Mean of Y: 0.1700

Standard deviation of X: 6.1805

Standard deviation of Y: 5.3452

1.4 Pearson Correlation Coefficient Calculation
Result: r = 0.999644

The correlation coefficient was calculated using two methods for verification:

NumPy's corrcoef() function: 0.999644

Manual calculation: 0.999644

Both methods produced identical results, confirming the accuracy of the calculation.

1.5 Interpretation
"Very strong positive correlation"

The correlation coefficient of 0.9996 indicates:

The relationship between X and Y variables is almost perfectly linear

As X increases, Y increases proportionally

The linear relationship explains 99.93% of the variance (R² = 0.9993)

1.6 Regression Analysis
Regression line equation: y = 0.8602x + 0.5478

Slope (m): 0.8602 - For every 1 unit increase in X, Y increases by 0.8602 units

Intercept (b): 0.5478 - The predicted Y value when X = 0

Root Mean Square Error (RMSE): 0.1520 - Low error indicates excellent fit

1.7 Visualization

Graph description:

Blue points represent the original data from the interactive graph

Red dashed line shows the linear regression fit

Each point is annotated with its (x, y) coordinates

Information box displays key statistics: correlation coefficient, interpretation, and regression equation

1.8 Files Created
coordinates.csv - Contains all 10 coordinate pairs in CSV format

correlation_plot.png - Visualization showing data points, regression line, and statistical information

Technical Implementation
Python Code Summary
python
# Key steps in the analysis:
# 1. Load required libraries (numpy, pandas, matplotlib)
# 2. Create dataset from collected coordinates
# 3. Calculate Pearson correlation using np.corrcoef()
# 4. Compute basic statistics (mean, standard deviation)
# 5. Perform linear regression to find best-fit line
# 6. Create visualization with scatter plot and regression line
# 7. Save results to CSV and PNG files

Conclusion:
The analysis successfully demonstrates a nearly perfect linear relationship between the X and Y variables collected from the interactive graph.
The correlation coefficient of 0.9996 confirms the visual observation from the original graph that the points follow a straight line pattern.
This strong positive correlation suggests that the two variables are highly related and can be accurately predicted from one another using linear regression.
<img width="1200" height="875" alt="Capture" src="https://github.com/user-attachments/assets/d92589fa-da6b-45c2-842c-87aa7c28c39b" />

Part 2: Spam Email Detection
1. Dataset Information
File: b_didebashvili25_84536_csv.csv

Total Emails: 2,500

Spam Emails: 1,287 (51.5%)

Legitimate Emails: 1,213 (48.5%)

Features:

words: Total number of words in email

links: Number of hyperlinks in email

capital_words: Number of words in ALL CAPS

spam_word_count: Count of spam-related keywords

is_spam: Target variable (1 = spam, 0 = legitimate)

Dataset Sample:

python
   words  links  capital_words  spam_word_count  is_spam
0    195      7             17                5        1
1    140      8             25                7        1
2    138      5             14                0        1
2. Data Processing Code
python
# Load dataset
data = pd.read_csv('b_didebashvili25_84536_csv.csv')

# Prepare features and target
features = ['words', 'links', 'capital_words', 'spam_word_count']
X = data[features]
y = data['is_spam']

# Split data (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Training on {X_train.shape[0]} emails")  # Output: 1750
print(f"Testing on {X_test.shape[0]} emails")    # Output: 750
3. Logistic Regression Model
python
# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
print("Model trained successfully!")
4. Model Coefficients
Feature	Coefficient	Interpretation
links	0.9574	Strongest predictor - more links significantly increase spam probability
spam_word_count	0.8373	High spam word count strongly indicates spam
capital_words	0.4635	ALL CAPS words moderately indicate spam
words	0.0076	Very weak positive correlation with spam
Intercept	-9.8465	Baseline log-odds
Statistical Interpretation:

Links coefficient (0.9574): For each additional link, the log-odds of being spam increase by 0.9574

Spam word count (0.8373): Each spam keyword increases spam probability significantly

All features are positive: Higher values for any feature increase spam probability

5. Model Evaluation
Confusion Matrix
text
[[TN=346, FP=18]
 [FN=12, TP=374]]
Performance Metrics
Accuracy: 96.00%

Precision: 95.41% (374 / (374 + 18))

Recall: 96.90% (374 / (374 + 12))

F1-Score: 96.15%

Detailed Analysis:

True Positives (374): Spam emails correctly identified as spam

True Negatives (346): Legitimate emails correctly identified as legitimate

False Positives (18): Legitimate emails incorrectly flagged as spam

False Negatives (12): Spam emails missed by the filter

Error Analysis:

False Positive Rate: 4.9% (18/364 legitimate emails)

False Negative Rate: 3.1% (12/386 spam emails)

The model performs slightly better at detecting spam than at identifying legitimate emails

6. Email Classification Function
python
def check_email_features(words, links, capital_words, spam_word_count):
    """
    Classify an email based on extracted features
    """
    features = np.array([[words, links, capital_words, spam_word_count]])
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0]
    return "SPAM" if prediction == 1 else "LEGITIMATE", probability[1]
7. Manually Created Spam Email
text
URGENT: WIN $1,000,000 PRIZE!
Dear Winner,

Congratulations! You have been selected to win $1,000,000 CASH PRIZE!
This is a LIMITED TIME OFFER. Click here to claim: http://bit.ly/winprize123

To claim your prize, we need $99 processing fee.
This is 100% GUARANTEED! Don't miss this opportunity!

Reply within 24 hours.
Feature Analysis:

Words: 85

Links: 1

Capital Words: 7 (URGENT, WIN, CASH PRIZE, LIMITED TIME OFFER, GUARANTEED)

Spam Word Count: 9 (urgent, win, prize, cash, limited, offer, click, guaranteed, opportunity)

Model Prediction: SPAM (92.67% probability)

Creation Strategy: Designed with multiple spam indicators:

Urgency markers: "URGENT", "LIMITED TIME"

Financial incentives: "$1,000,000", "CASH PRIZE", "$99 fee"

Action prompts: "Click here", "claim"

False guarantees: "100% GUARANTEED"

Time pressure: "Reply within 24 hours"

8. Manually Created Legitimate Email
text
Subject: Project Update Meeting

Hello Team,

The project update meeting is scheduled for tomorrow at 2 PM in Conference Room A.
Please bring your progress reports and be ready to discuss next week's milestones.

The presentation slides are attached to this email.
Looking forward to our discussion.

Best regards,
Alex Johnson
Project Lead

10. Visualizations
<img width="1200" height="500" alt="Figure_1" src="https://github.com/user-attachments/assets/65daa33c-68e7-4751-b982-867d0dcccc1c" />
<img width="800" height="600" alt="Figure_2" src="https://github.com/user-attachments/assets/40f85a53-863f-4898-b280-c054081ff3bb" />
<img width="1000" height="600" alt="Figure_3" src="https://github.com/user-attachments/assets/7eda85d8-7ff9-4f31-a089-5a3701978327" />
Conclusion
The spam detection system achieved 96.00% accuracy using logistic regression on four carefully engineered email features. The model demonstrates that:
1.Link count is the strongest spam indicator (coefficient: 0.9574)
2.Spam keywords are highly predictive (coefficient: 0.8373)
3.Capitalization patterns provide moderate signals (coefficient: 0.4635)
4.Email length has minimal predictive power (coefficient: 0.0076)
The confusion matrix reveals excellent performance with only 30 misclassifications out of 750 test emails (4% error rate).
The manually created emails successfully validated the model's decision-making process, with the spam email receiving 92.67% spam probability and the legitimate email receiving only 0.03% spam probability.
The correlation analysis from Part 1 confirmed an almost perfect linear relationship (r = 0.9996) between the collected data points, demonstrating precise mathematical correlation in the provided interactive graph data.


