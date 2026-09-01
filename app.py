import streamlit as st
import pandas as pd
import joblib


# --------------------------------
# Page Configuration
# --------------------------------

st.set_page_config(
    page_title="Netflix Movie Recommender",
    page_icon="🎬",
    layout="wide"
)


# --------------------------------
# Load Dataset
# --------------------------------

df = pd.read_csv(
    "netflix_movies_cleaned.csv"
)


# --------------------------------
# Load ML Objects
# --------------------------------

tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)

tfidf_matrix = joblib.load(
    "tfidf_matrix.pkl"
)


# --------------------------------
# Movie Index
# --------------------------------

movie_indices = pd.Series(
    df.index,
    index=df["Title"]
).drop_duplicates()


# --------------------------------
# Recommendation Function
# --------------------------------

def recommend_movies(
    movie_title,
    num_recommendations
):

    idx = movie_indices[movie_title]

    similarity_scores = list(
        enumerate(cosine_sim[idx])
    )

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    similarity_scores = similarity_scores[
        1:num_recommendations + 1
    ]

    result = []

    for movie_index, score in similarity_scores:

        result.append({

            "Movie": df.iloc[movie_index]["Title"],

            "Genre": df.iloc[movie_index]["Genre"],

            "Director": df.iloc[movie_index]["Director"],

            "Cast": df.iloc[movie_index]["Cast"],

            "Country": df.iloc[movie_index]["Country"],

            "Language": df.iloc[movie_index]["Language"],

            "Rating": df.iloc[movie_index]["Rating"],

            "Similarity": round(score, 3)

        })

    return pd.DataFrame(result)


# --------------------------------
# Title
# --------------------------------

st.title(
    "🎬 Netflix Movie Recommendation System"
)

st.write(
    "Find movies similar to your favorite movie."
)


# --------------------------------
# Sidebar
# --------------------------------

st.sidebar.header(
    "Recommendation Settings"
)

num_recommendations = st.sidebar.slider(
    "Number of Recommendations",
    min_value=1,
    max_value=10,
    value=5
)


# --------------------------------
# Movie Selection
# --------------------------------

movie_title = st.selectbox(
    "Select a Movie",
    df["Title"].values
)


# --------------------------------
# Recommendation Button
# --------------------------------

if st.button(
    "🔍 Recommend Movies"
):

    recommendations = recommend_movies(
        movie_title,
        num_recommendations
    )

    st.subheader(
        "🎥 Recommended Movies"
    )

    for _, movie in recommendations.iterrows():

        st.markdown(
            f"### 🎬 {movie['Movie']}"
        )

        st.write(
            f"**Genre:** {movie['Genre']}"
        )

        st.write(
            f"**Director:** {movie['Director']}"
        )

        st.write(
            f"**Cast:** {movie['Cast']}"
        )

        st.write(
            f"**Country:** {movie['Country']}"
        )

        st.write(
            f"**Language:** {movie['Language']}"
        )

        st.write(
            f"**Rating:** {movie['Rating']}"
        )

        st.write(
            f"**Similarity Score:** "
            f"{movie['Similarity']}"
        )

        st.divider()
