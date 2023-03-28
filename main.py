import streamlit as st
import pandas as pd
from PIL import Image
import requests
import json
import streamlit as st
from streamlit.components.v1 import html
import seaborn as sns
import pandas as pd
import numpy as np
from htbuilder import HtmlElement, div, hr, a, p, img, styles
from htbuilder.units import percent, px
from PIL import Image
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as mt


image_spotify = Image.open('images/spotify.png')
st.image(image_spotify, width=300)


st.title("US Spotify Charts from 2017-2022")

st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('Select Page',['Summary 🚀','Visualization 📊','Prediction 📈'])
def img_to_bytes(img_path):
    img_bytes = Path(img_path).read_bytes()
    encoded = base64.b64encode(img_bytes).decode()
    return encoded



def main():
    def _max_width_():
        max_width_str = f"max-width: 1000px;"
        st.markdown(
            f"""
        <style>
        .reportview-container .main .block-container{{
            {max_width_str}
        }}
        </style>
        """,
            unsafe_allow_html=True,
        )


    # Hide the Streamlit header and footer
    def hide_header_footer():
        hide_streamlit_style = """
                    <style>
                    footer {visibility: hidden;}
                    </style>
                    """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # increases the width of the text and tables/figures
    _max_width_()

    # hide the footer
    hide_header_footer()

if app_mode == 'Summary 🚀':
    st.subheader("01 Summary Page - Spotify Data Analysis 🚀")
    st.markdown("##### Objectives")
    st.markdown("Spotify is a popular music streaming platform used by hundreds of millions of users each month. We recognize it's importance in demostrating popularity of music today and wondered if we could use the most popular songs on Spotify from two different charts to better understand modern music trends and predict fucture ones in the United States.")

    df = pd.read_csv("df_spotify_final.csv")

    num = st.number_input('No. of Rows', 5, 10)

    head = st.radio('View from top (head) or bottom (tail)', ('Head', 'Tail'))
    if head == 'Head': 
        st.dataframe(df.head(num))
    else:
        st.dataframe(df.tail(num))

    st.text('(Rows,Columns)')
    st.write(df.shape)

    st.markdown("##### Key Variables")
    st.markdown("- Rank: Current Place on Top 200 list")
    st.markdown("- Best Rank: The highest rank of any given song")
    st.markdown("- Date: Specific date at the time of ranking")
    st.markdown("- Title: Title of song")
    st.markdown("- URL: Unique Spotify URL of song")
    st.markdown("- Artist: Singer/ Musician/ Producer of song")
    st.markdown("- Trend: Whether the song moved up, moved down, stayed in the same position, or was a new entry at the time")
    st.markdown("- Streams: Number of streams on the platform at the time of ranking")
    st.markdown("- Consecutive Weeks on Top 200: The number of consecutive release cycles a song stayed on the chart")

    st.markdown("### Description of Data")
    st.dataframe(df.describe())
    st.markdown("Descriptions for all quantitative data **(rank and streams)** by:")

    st.markdown("Count")
    st.markdown("Mean")
    st.markdown("Standard Deviation")
    st.markdown("Minimum")
    st.markdown("Quartiles")
    st.markdown("Maximum")

    st.markdown("### Missing Values")
    st.markdown("Null or NaN values.")

    dfnull = df.isnull().sum()/len(df)*100
    totalmiss = dfnull.sum().round(2)
    st.write("Percentage of total missing values:",totalmiss)
    st.write(dfnull)
    if totalmiss <= 30:
        st.success("We have less then 30 percent of missing values, which is good. This provides us with more accurate data as the null values will not significantly affect the outcomes of our conclusions. And no bias will steer towards misleading results. ")
    else:
        st.warning("Poor data quality due to greater than 30 percent of missing value.")
        st.markdown(" > Theoretically, 25 to 30 percent is the maximum missing values are allowed, there's no hard and fast rule to decide this threshold. It can vary from problem to problem.")

    st.markdown("### Completeness")
    st.markdown(" The ratio of non-missing values to total records in dataset and how comprehensive the data is.")

    st.write("Total data length:", len(df))
    nonmissing = (df.notnull().sum().round(2))
    completeness= round(sum(nonmissing)/len(df),2)

    st.write("Completeness ratio:",completeness)
    st.write(nonmissing)
    if completeness >= 0.80:
        st.success("We have completeness ratio greater than 0.85, which is good. It shows that the vast majority of the data is available for us to use and analyze. ")    
    else:
        st.success("Poor data quality due to low completeness ratio( less than 0.85).")

if app_mode == 'Visualization 📊':
    st.subheader("02 Visualization Page - Spotify Data Analysis 📊")
    #response = requests.get("https://lookerstudio.google.com/u/0/reporting/97c91d4e-e116-488f-b695-50179f0c7a11/page/pYAKD")
    #html_code = response.text
    #st.write("My Looker Dashboard")
    #html_code = f'<iframe srcdoc="{html_code}" width="100%" height="600" frameborder="0"></iframe>'
    #html(html_code)

    st.markdown("[![Foo](https://i.postimg.cc/HLh9Twzf/Screenshot-2023-03-28-at-12-59-19.png)](https://lookerstudio.google.com/u/0/reporting/97c91d4e-e116-488f-b695-50179f0c7a11/page/pYAKD)")

    #image_dashboard = Image.open('images/dashboard.png')
    #st.image(image_dashboard)


if app_mode == 'Prediction 📈':



    ### The st.title() function sets the title of the Streamlit application to "Mid Term Template - 03 Prediction Page 🧪".
    st.subheader("03 Prediction Page - Spotify Data Analysis 🧪")

    ### The pd.read_csv() function loads a CSV file of wine quality data into a Pandas DataFrame called "df".
    pre_df = pd.read_csv("spotify_final.csv")
    df = pre_df
    df['Streams'] = pre_df['Streams'].str.replace(',','').astype(int)
    df['Streams increase since last release'] = pre_df['Streams increase since last release'].str.replace(',','').astype(int)


    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    list_variables = df[[ 'Rank', 'Streams',
       'Streams increase since last release', 'Weeks in top 200',
       'Consecutive weeks in top 200', 'Best rank', 'Prior rank',
       'Rank change', 'Up or down', 'Song rank in next top 200 release',
       'Steams in next release', 'In next release']].columns
    select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.
    train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)

    new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    list_var = new_df.columns

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    output_multi = st.multiselect("Select Explanatory Variables", list_var,default=['Streams','Streams increase since last release'])

    new_df2 = new_df[output_multi]
    x =  new_df2
    y = df[select_variable]

    ### The train_test_split() function splits the data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)

    ### The LinearRegression() function creates a linear regression model.
    lm = LinearRegression()

    ### The lm.fit() function fits the linear regression model to the training data.
    lm.fit(X_train,y_train)

    ###The lm.predict() function generates predictions for the testing data.
    predictions = lm.predict(X_test)

    ### The st.columns() function creates two columns to display the feature columns and target column.
    col1,col2 = st.columns(2)
    col1.subheader("Feature Columns top 25")
    col1.write(x.head(25))
    col2.subheader("Target Column top 25")
    col2.write(y.head(25))

    ### The st.subheader() function creates a subheading for the results section.
    st.subheader('🎯 Results')

    ### The st.write() function displays various metrics for the linear regression model, including the variance explained, mean absolute error, mean squared error, and R-squared score. The results are rounded to two decimal places using the np.round() function.
    st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))




    #### The st.title() function sets the title of the Streamlit application to "Mid Term Template - 03 Prediction Page 🧪".
    #st.title("Spotify Data Analysis - 03 Prediction Page 🧪")

    ### The pd.read_csv() function loads a CSV file of wine quality data into a Pandas DataFrame called "df".
    #df = pd.read_csv("df_spotify.csv")
    #list_new_chart = []
    #for i in range(0,len(df["chart"])):
    #    if "50" in df["chart"][i]:
    #        list_new_chart.append(50)
    #    else:
    #        list_new_chart.append(200)

    #df["chart"] = list_new_chart
    #df_new = df[["rank","chart","streams"]]
    #df_new_2 = df_new.dropna()
    #df_new_2.head(3)

    ### The st.sidebar.selectbox() function creates a dropdown menu in the sidebar that allows users to select the target variable to predict.
    #list_variables = df_new_2.columns
    #select_variable =  st.sidebar.selectbox('🎯 Select Variable to Predict',list_variables)

    ### The st.sidebar.number_input() function creates a number input widget in the sidebar that allows users to select the size of the training set.
    #train_size = st.sidebar.number_input("Train Set Size", min_value=0.00, step=0.01, max_value=1.00, value=0.70)

    #new_df= df.drop(labels=select_variable, axis=1)  #axis=1 means we drop data by columns
    #list_var = df_new_2.columns

    ### The st.multiselect() function creates a multiselect dropdown menu that allows users to select the explanatory variables.
    #output_multi = st.multiselect("Select Explanatory Variables", list_var,default=['streams','rank'])

    #new_df2 = df_new_2[output_multi]
    #x =  new_df2
    #y = df_new_2[select_variable]

    ### The train_test_split() function splits the data into training and testing sets.
    #X_train, X_test, y_train, y_test = train_test_split(x,y,test_size=train_size)

    ### The LinearRegression() function creates a linear regression model.
    #lm = LinearRegression()

    ### The lm.fit() function fits the linear regression model to the training data.
    #lm.fit(X_train,y_train)

    ###The lm.predict() function generates predictions for the testing data.
    #predictions = lm.predict(X_test)

    ### The st.columns() function creates two columns to display the feature columns and target column.
    #col1,col2 = st.columns(2)
    #col1.subheader("Feature Columns top 25")
    #col1.write(x.head(25))
    #col2.subheader("Target Column top 25")
    #col2.write(y.head(25))

    ### The st.subheader() function creates a subheading for the results section.
    #st.subheader('🎯 Results')

    ### The st.write() function displays various metrics for the linear regression model, including the variance explained, mean absolute error, mean squared error, and R-squared score. The results are rounded to two decimal places using the np.round() function.
    #st.write("1) The model explains,", np.round(mt.explained_variance_score(y_test, predictions)*100,2),"% variance of the target feature")
    #st.write("2) The Mean Absolute Error of model is:", np.round(mt.mean_absolute_error(y_test, predictions ),2))
    #st.write("3) MSE: ", np.round(mt.mean_squared_error(y_test, predictions),2))
    #st.write("4) The R-Square score of the model is " , np.round(mt.r2_score(y_test, predictions),2))
if __name__=='__main__':
    main()

#st.markdown(" ")
#st.markdown("### 👨🏼‍💻 **App Contributors:** ")
#st.image(['images/gaetan.png'], width=100,caption=["Gaëtan Brison"])

#st.markdown(f"####  Link to Project Website [here]({'https://github.com/NYU-DS-4-Everyone/Linear-Regression-App'}) 🚀 ")
#st.markdown(f"####  Feel free to contribute to the app and give a ⭐️")


def image(src_as_string, **style):
    return img(src=src_as_string, style=styles(**style))


def link(link, text, **style):
    return a(_href=link, _target="_blank", style=styles(**style))(text)


def layout(*args):

    style = """
    <style>
      # MainMenu {visibility: hidden;}
      footer {visibility: hidden;background - color: white}
     .stApp { bottom: 80px; }
    </style>
    """
    style_div = styles(
        position="fixed",
        left=0,
        bottom=0,
        margin=px(0, 0, 0, 0),
        width=percent(100),
        color="black",
        text_align="center",
        height="auto",
        opacity=1,

    )

    style_hr = styles(
        display="block",
        margin=px(8, 8, "auto", "auto"),
        border_style="inset",
        border_width=px(2)
    )

    body = p()
    foot = div(
        style=style_div
    )(
        hr(
            style=style_hr
        ),
        body
    )

    st.markdown(style, unsafe_allow_html=True)

    for arg in args:
        if isinstance(arg, str):
            body(arg)

        elif isinstance(arg, HtmlElement):
            body(arg)

    st.markdown(str(foot), unsafe_allow_html=True)

def footer2():
    myargs = [
        "👩‍💻 Made by ",
        link("https://github.com/NYU-DS-4-Everyone", "NYU students - Lia Cociorva & Aisha Njau"),
        " ⭐️"
    ]
    layout(*myargs)


if __name__ == "__main__":
    footer2()