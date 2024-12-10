###
# About the Final Project

#For the final project, you will identify a Supervised Learning problem to perform EDA and model analysis.  The project has 140 total points. In the instructions is a summary of the criteria you will use to guide your submission and review others’ submissions.   You will submit three deliverables:
#Deliverable 1

#A Jupyter notebook showing a supervised learning problem description, EDA procedure, analysis (model building and training), result, and discussion/conclusion.

#Suppose your work becomes so large that it doesn’t fit into one notebook (or you think it will be less readable by having one large notebook). In that case, you can make several notebooks or scripts in a GitHub repository (as deliverable 3) and submit a report-style notebook or pdf instead.

#If your project doesn't fit into Jupyter notebook format (E.g. you built an app that uses ML), write your approach as a report and submit it in a pdf form.
#Deliverable 2

#A video presentation or demo of your work. The presentation should be a condensed version as if you're doing a short pitch to advertise your work, so please focus on the highlights:

#    What problem do you solve?

#    What ML approach do you use, or what methods does your app use?

#    Show the result or run an app demo.

#The minimum video length is 5 min, the maximum length is 15 min. The recommended length is about 10 min. Submit the video in .mp4 format.
#Deliverable 3

#A public project GitHub repository with your work (please also include the GitHub repo URL in your notebook/report and slides).
#Data byproduct

#If your project creates data and you want to share it, an excellent way to share would be through a Kaggle dataset or similar. Similarly, suppose you want to make your video public. In that case, we recommend uploading it to YouTube or similar and posting the link(s) to your repository or blog instead of a direct upload to GitHub.
#It is generally a good practice not to upload big files to a git repository.
###



import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

df = pd.read_csv('data/adult.csv')
print('Shape of dataset (Number of observations, number of features):', df.shape)
print(df.head())
print(df.info())
print(df.describe())

#Create a mask showing true in rows containing '?'
mask = df.map(lambda x: '?' in str(x))
#Drop rows from the mask
df = df[~mask.any(axis=1)]
#Drop duplicated rows
df = df.drop_duplicates()
#Drop the fnlwgt column
df = df.drop(columns=['fnlwgt'])
#Replace '.' with '_' in column names
df = df.rename(columns={'education.num':'education_num',
                        'marital.status':'marital_status',
                        'capital.gain':'capital_gain',
                        'capital.loss':'capital_loss',
                        'hours.per.week':'hours_per_week',
                        'native.country':'native_country'})
#Using pie charts to display distributions of categorical variables
cat_df = df.dtypes[df.dtypes == 'object']
for column in list(cat_df.index):
    counts = df[column].value_counts()

    plt.figure(figsize=(10,10))
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=180, colors=plt.cm.Paired.colors)
    plt.title(f'Distribution of {column}')
    plt.show()

#Using violin plots to display distributions of numeric variables
num_df = df.dtypes[df.dtypes == 'int64']
for column in list(num_df.index):
    plt.figure(figsize=(10,10))
    sns.violinplot(y=df[column])
    plt.title(f'Distribution of {column}')

#Looking at correlation matrix between numeric variables
numeric_df = df.select_dtypes(include=['number'])
correlation_matrix = numeric_df.corr()
plt.figure(figsize=(12, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='magma', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix')

#Binarizing labels in categorical variables
label_encoder = LabelEncoder()
categorical_columns = list(cat_df.index)
df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)

#Create testing and training sets
y = df['income']
X = df.drop(columns='income')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#Trying initial fit
nlsvm = SVC(C=1, kernel='rbf', gamma=1)
nlsvm.fit(X_train, y_train)
scores = cross_val_score(nlsvm, X_test, y_test)
print("cross-val mean-accuracy: {:.3f}".format(np.mean(scores)))

#Grid Search to find optimal C and gamma
ranges = {'C': np.logspace(-5,5, num=20, base=2), 'gamma': np.logspace(-5,5, num=20, base=2)}
model = SVC(kernel='rbf')
grid = GridSearchCV(model, ranges, cv=3, scoring='accuracy').fit(X_train, y_train)

#Visualizing Grid Search
def plotSearchGrid(grid):
    scores = [x for x in grid.cv_results_["mean_test_score"]]
    scores = np.array(scores).reshape(len(grid.param_grid["C"]), len(grid.param_grid["gamma"]))

    plt.figure(figsize=(10, 8))
    plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
    plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
               norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
    plt.xlabel('gamma')
    plt.ylabel('C')
    plt.colorbar()
    plt.xticks(np.arange(len(grid.param_grid["gamma"])), grid.param_grid["gamma"], rotation=45)
    plt.yticks(np.arange(len(grid.param_grid["C"])), grid.param_grid["C"])
    plt.title('Validation accuracy')
    plt.show()


class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


from IPython.core.display import HTML

HTML("""
<style>
.MathJax nobr>span.math>span{border-left-width:0 !important};
</style>
""")
plotSearchGrid(grid)
print("Best parameters: ", grid.best_params_)
print("Best cross-validation accuracy: ", grid.best_score_)

#Comparing to Logistic Regression
LogReg = LogisticRegression(class_weight='balanced', solver='liblinear').fit(X_train, y_train)
scores = cross_val_score(LogReg, X_test, y_test)
print("cross-val mean-accuracy: {:.3f}".format(np.mean(scores)))

print('Done!')
