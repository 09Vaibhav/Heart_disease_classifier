import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, StratifiedKFold, StratifiedShuffleSplit

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve
from sklearn.metrics import precision_score, recall_score
from PIL import Image


def main():
    st.title("Binary Classification Web App")
    st.sidebar.title("Binary Classification Web App")
    st.markdown("Are you at a risk of heart attack? ")
    st.sidebar.markdown("Are you at a risk of heart attack? ")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    image = Image.open('Heart-Attack-Symptoms.jpg')
    st.image(image,'Heart  Attack')

    @st.cache_data(persist=True)
    def load_data():
        data = pd.read_csv(r'C:\\Users\\User\\Desktop\\glide_try\\heart.csv')
        
        return data
    
    @st.cache_data(persist=True)
    def split(df):
        numeric_data = ["age", "trtbps","chol","thalachh","oldpeak"]
        df_numeric = df.loc[:, numeric_data]

        for i in numeric_data:
            # IQR
            Q1 = np.percentile(df.loc[:, i],25)
            Q3 = np.percentile(df.loc[:, i],75)
            
            IQR = Q3 - Q1
            
            print("Shape with outliers: ", df.loc[:, i].shape)
            
            # upper bound
            upper = np.where(df.loc[:, i] >= (Q3 +2.5*IQR))
            
            # lower bound
            lower = np.where(df.loc[:, i] <= (Q1 - 2.5*IQR))
            
            print("{} -- {}".format(upper, lower))
            
            try:
                df.drop(upper[0], inplace = True)
            except: print("KeyError: {} not found in axis".format(upper[0]))
            
            try:
                df.drop(lower[0], inplace = True)
            except:  print("KeyError: {} not found in axis".format(lower[0]))
        
        categorical_data = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]

        df1 = df.copy()
        df1 = pd.get_dummies(df1, columns = categorical_data[:-1], drop_first = True)
        # select X and y for model building
        X = df1.drop(["output"], axis = 1)
        y = df1[["output"]]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 3)

        return X_train, X_test, y_train, y_test
    
    def plot_metrics(metrics_list):
        if 'Confusion Matrix' in metrics_list:
            st.subheader("Confusion Matrix")
            plot_confusion_matrix(model, x_test, y_test, display_labels=class_names)
            st.pyplot()

        if 'ROC Curve' in metrics_list:
            st.subheader("ROC Curve")
            plot_roc_curve(model, x_test, y_test)
            st.pyplot()
        
        if 'Precision-Recall Curve' in metrics_list:
            st.subheader('Precision-Recall Curve')
            plot_precision_recall_curve(model, x_test, y_test)
            st.pyplot()
    
    
    df = load_data()
    class_names = ['No Risk ', 'High Risk']
    st.sidebar.subheader('Analyze Data')
    Analyzer = st.sidebar.selectbox("Analyzer", ("Categorical Data Analysis", "Numerical Data Analysis", "Box plot","Correlation matrix"))
    if Analyzer == 'Categorical Data Analysis':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Categorical Data Analysis Results")
            categorical_data = ["sex", "cp","fbs","restecg","exng","slp","caa","thall","output"]
            df_categoric = df.loc[:, categorical_data]
            for i in categorical_data:
                fig =plt.figure()
                sns.countplot(x = i, data = df_categoric, hue = "output")
                st.pyplot(fig)
                plt.title(i)
                if (i == 'sex'):
                    st.markdown(
                        """
                        Observations :
                        - The given dataset have 68 percent males  and 32 percent females in it.
                        - 32 percent of total male is at risk of having a heart disease.
                        - 23 percent of total female is at risk of having a heart disease.
                        """
                        )
                if (i == 'cp'):
                    st.markdown(
                        """
                        Observations :
                        - Most of the people have type 0 (asymptomatic) chest pain and have less than 30 percent probability of having a heart disease .
                        - Those having type 1 (typical angina) or type 2 (atypical angina) chest pain have more than 75 percent chances of having a heart disease .
                        - symptoms for type 1 cp are chest , arm and jaw pain.
                        - symptoms for type 2 cp are epigastric , back pain , burning sensation and stabbing.
                        - 10 percent of people have type 3 (non anginal) chest pain.
                        - angina is the chest pain or discomfort that happens when heart is not receiving enough oxygen rich blood.
                        """
                        )
                if (i == 'fbs'):
                        st.markdown(
                        """
                        Observations :
                        - Most of the people (86 percent) have fbs of type 0 (fbs < 120 mg/dL) .
                        - 14 percent of people have fbs of type 1 (fbs > 120 mg/dL) 
                        - High fbs (fasting blood sugar > 120 mg/dL) indicates that a person have high chance of diabetes .
                        """
                        )
                if (i == 'restecg'):
                        st.markdown(
                        """
                        Observations :
                        - Most of the people have either type 0 (hypertrophy) or type 1 (normal) ecg result.
                        - people with normal ecg result is having more than 65 percent chances of having a heart disease.
                        - Very less number of people have type 2 (having ST-T wave abnormality) restecg  .
                        """
                        )
                if (i == 'exng'):
                        st.markdown(
                        """
                        Observations :
                        - people with type 0 (angina which are not exercise induced) exng have 70 percent chances of having a heart disease.
                        - people with type 1 (exercise induced angina) have less probability of having a heart disease.
                        - exercise induced angina usually happens during activity(exertion) and goes away with rest or angina medication.
                        """
                        )
                if (i == 'slp'):
                        st.markdown(
                        """
                        Observations :
                        - very less number of people have type 0 (down sloping) slp.
                        - people with type 1 (flat slope) slp have less probability of having a heart disease.
                        - people with type 2 (up sloping) have more than 70 percent chance of having a heart disease.
                        """
                        )
                if (i == 'caa'):
                        st.markdown(
                        """
                        Observations :
                        - caa indicates the number of major vessels.
                        - people with 0 major vessel have approximately 80 percent chance of having a heart disease.
                        - 1 out of 3 people with one major vessel have the chance of having a  heart disease. 
                        """
                        )
                if (i == 'thall'):
                        st.markdown(
                        """
                        Observations :
                        - the number of people having type 2 (normal blood flow) thall is twice that of having type 3 (reversible defect) thall.
                        - people having normal blood flow have more chances (75 percent) of having a heart disease.
                        - less than 10 percent of people have type 0 (fixed defect) thall .
                        """
                        )
                if (i == 'output'):
                        st.markdown(
                        """
                        Observations :
                        - 46 percent of the people is at low risk (type 0) of having a heart disease.
                        - 54 percent of the people is at high risk (type 1) of heart disease.
                        """
                        )
                    
    if Analyzer == 'Numerical Data Analysis':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Numerical Data Analysis Results")
            numeric_data = ["age", "trtbps","chol","thalachh","oldpeak","output"]
            df_numeric = df.loc[:, numeric_data]
            fig2 = sns.pairplot(df_numeric, hue = "output", diag_kind = "kde")
            st.pyplot(fig2)
            plt.show()
            st.markdown(
                 """
                 Abbreviations :
                 - trtbps : resting blood pressure (in mm Hg).
                 - chol : cholestrol in mg/dl fetched via BMI sensor
                 - thalach : maximum heart rate achieved
                 """
                 )
            st.markdown(
                 """
                 Observations :
                 - cholestrol and age are positively co-related for both 0(less than 50 percent chances of having heart disease) and 1 output(more than 50 percent chances of having heart disease).
                 - thalach (maximum heart rate) and age are negatively co-related for both o and 1 output .
                 - oldpeak and chol (cholestrol) are right skewed (positive skewed) .
                 - in right skewed distribution, the mean is greater than the median . so, the mean overestimates the most common values.
                 - thalach(maximum heart rate achieved) is  left skewed (negative skewed).
                 - in left skewed distribution, the mean is less than the median . so, the mean underestimates the most common values.(hence median is used instead of mean)
                 """
                 )
    
    if Analyzer == 'Box plot':
        scaler = StandardScaler()
        numeric_data = ["age", "trtbps","chol","thalachh","oldpeak","output"]
        scaled_data = scaler.fit_transform(df[numeric_data[:-1]])
        # scaled_data
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Box and whisker plot Results")
            df_dummy = pd.DataFrame(scaled_data, columns = numeric_data[:-1])
            # df_dummy.head()
            df_dummy = pd.concat([df_dummy, df.loc[:, "output"]], axis = 1)
            # df_dummy.head()
            data_melted = pd.melt(df_dummy, id_vars = "output", var_name = "features", value_name = "value")
            # data_melted.head(20)
            # box plot
            fig3 =plt.figure()
            sns.boxplot(x = "features", y = "value", hue = "output", data= data_melted)
            st.pyplot(fig3)
            plt.show()
            st.markdown(
                 """
                 Observations :
                 - Box and whisker plot is one of the best way to check for outliers. we can even get the distribution (normal , right or left skewed) type of data using box plot.
                 - the points outside the whiskers are the outliers.here, all the five numerical features shows the presence of outliers .
                 - the presence of skewness is denoted by the shift of the median (centre line of box) towards either side of the extremas.
                 """
                 )
    
    if Analyzer == 'Correlation matrix':
        if st.sidebar.button("Analyze", key='Analyze'):
            st.subheader("Correlation matrix Results")
            fig4 =plt.figure(figsize = (14,10))
            sns.heatmap(df.corr(), annot = True, fmt = ".2f", linewidths = .7)
            st.pyplot(fig4)
            plt.show()
            st.markdown(
                 """
                 Observations :
                 - cp and thalach are top correlated features with output having 0.43 and 0.42 coefficient.
                 - exng and oldpeak are top negatively correlated features  with output having -0.44 and -0.43 coefficient.
                 - fbs and chol have almost negligible (very low) correlation with output.
                 - oldpeak and slp have high negative co-relation with each other with coefficient value as -0.58.
                 """
                 )



    x_train, x_test, y_train, y_test = split(df)

    st.sidebar.subheader("Choose Classifier")
    classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

    if classifier == 'Support Vector Machine (SVM)':
        st.sidebar.subheader("Model Hyperparameters")
        #choose parameters
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_SVM')
        kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key='kernel')
        gamma = st.sidebar.radio("Gamma (Kernel Coefficient)", ("scale", "auto"), key='gamma')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
        
        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Support Vector Machine (SVM) Results")
            model = SVC(C=C, kernel=kernel, gamma=gamma)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Logistic Regression':
        st.sidebar.subheader("Model Hyperparameters")
        C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key='C_LR')
        max_iter = st.sidebar.slider("Maximum number of iterations", 100, 500, key='max_iter')

        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Logistic Regression Results")
            model = LogisticRegression(C=C, penalty='l2', max_iter=max_iter)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if classifier == 'Random Forest':
        st.sidebar.subheader("Model Hyperparameters")
        n_estimators = st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key='n_estimators')
        max_depth = st.sidebar.number_input("The maximum depth of the tree", 1, 20, step=1, key='max_depth')
        bootstrap = st.sidebar.radio("Bootstrap samples when building trees", ('True', 'False'), key='bootstrap')
        metrics = st.sidebar.multiselect("What metrics to plot?", ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))

        if st.sidebar.button("Classify", key='classify'):
            st.subheader("Random Forest Results")
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap=bootstrap, n_jobs=-1)
            model.fit(x_train, y_train)
            accuracy = model.score(x_test, y_test)
            y_pred = model.predict(x_test)
            st.write("Accuracy: ", accuracy.round(2))
            st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
            st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
            plot_metrics(metrics)
    
    if st.sidebar.checkbox("Show raw data", False):
        st.subheader("Heart Attack Prediction Data Set (Classification)")
        st.write(df)
        st.markdown("This [data set](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset) includes various attributes , categorical (such as sex,chest pain type ,fasting blood sugar level etc) and nummerical (such as age,cholestrol, oldpeak etc) that will predict the risk of heart attack  ) ) "
        "Heart attack is a medical emergency and is very common with approximately 10 million case per year (in india))"
        "It occurs when the flow of the blood to the heart is sevverely  reduced  or blocked")

        st.subheader("Data Variable Description")
        st.markdown(" 1. age -age in years")
        st.markdown("2. sex - (1 = male, 0 = female) ")
        st.markdown(
            """
            3. cp - chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 0 = asymptomatic):
                - typical angina : symptoms are chest, arm, jaw pain
                - atypical angina : symptoms are epigastric , back pain, burning, stabbing
                - non anginal pain : people having chest pain without heart disease. they also suffer from panic disorder, anxiety, depression
            """
            )
        st.markdown("4. trestbps - resting blood pressure (in mm Hg on admission to the hospital)")
        st.markdown("5. chol - serum cholestoral in mg/dl")
        st.markdown(
            """
            6. fbs - fasting blood sugar > 120 mg/dl (1 = true; 0 = false):
                - it indicates blood sugar level after an overnight fast.
                - fbs < 120mg/dL is normal and fbs > 120 indicates patient have diabetes
            """
            )
        st.markdown(
            """
            7. rest_ecg - resting electrocardiographic results (1 = normal; 2 = having ST-T wave abnormality; 0 = hypertrophy):
                - hypertrophy : inndicates the thickening of wall of heart's main pumping chamber.
                - having ST-T wave abnormality : indicates the blockage of the main artery.
            """
            )
        st.markdown("8. thalach - maximum heart rate achieved")
        st.markdown(
            """
            9. exang - exercise induced angina (1 = yes; 0 = no):
                - exercise induced angina are temporary and goes away with rest.
            """
            )
        st.markdown("10. oldpeak - ST depression induced by exercise relative to rest")
        st.markdown("11. slope - the slope of the peak exercise ST segment (2 = upsloping; 1 = flat; 0 = downsloping)")
        st.markdown(
            """
            12. ca - number of major vessels (0-3) colored by flourosopy
                - major vessels are the blood vessels that are directly connected to heart.
                - eg : pulmonary artery, pulmonary veins , vena cava, aorta
            """
            )
        st.markdown(
            """
            13. thal - (2 = normal; 1 = fixed defect; 3 = reversable defect) :
                - Blood disorder called thalassemia.
                - normal: indicates normal blood flow.
                - fixed defects : no blood flow in some parts of the heart.
                - reversible defects : blood flow is observed but it is not normal.
            """
            )
        st.markdown("14. num - the predicted attribute - diagnosis of heart disease (angiographic disease status) (Value 0 = < diameter narrowing; Value 1 = > 50% diameter narrowing)")
        st.markdown("The reference for the above description are taken from  [kaggle](https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset/discussion/234843) and [National Library of Medicine](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4000924/)")

        

if __name__ == '__main__':
    main()


