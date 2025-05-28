import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Set the page config
st.set_page_config(layout="wide")

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Custom font and UI styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }
    .sidebar .sidebar-content {
        background-color: #07B1FC !important;
        color: white;
    }
    .stButton>button {
        background-color: #06516F !important;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0098DB !important;
        color: white;
    }
    .stNumberInput input {
        border: 2px solid #06516F !important;
    }
    .stNumberInput input:focus {
        border: 2px solid #0098DB !important;
    }
    </style>
""", unsafe_allow_html=True)

def preprocess_text(text, excluded):
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    try:
        tokens = word_tokenize(str(text).lower())
    except:
        return []

    custom_stopwords = set(stopwords.words('english')).union({
        'and', 'or', 'but', 'if', 'also', 'yhis', 'yrs', 'because', 'ca', 'would', 'let',
        'abt', 'ac', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'with', 'is', 'are',
        'was', 'were', 'of', 'for'
    }).union(set(excluded))

    return [word for word in tokens if word.isalnum() and word not in custom_stopwords]


# Sentiment analysis logic
def sentiment_analysis(text):
    return TextBlob(text).sentiment.polarity

# Emoji-based sentiment icon
def sentiment_icon(sentiment):
    if sentiment > 0.1:
        return "😊"
    elif sentiment < -0.1:
        return "😞"
    else:
        return "😐"

# Graph creation function
def create_network_graph(reviews_tokens, keyword=None, min_occurrence=1):
    G = nx.Graph()
    word_counts = {}
    word_sentiments = {}

    for tokens, sentiment in reviews_tokens:
        for word in tokens:
            word_counts[word] = word_counts.get(word, 0) + 1
            word_sentiments.setdefault(word, []).append(sentiment)

    for tokens, sentiment in reviews_tokens:
        for i in range(len(tokens)):
            for j in range(i + 1, len(tokens)):
                if (keyword == "All" or tokens[i] == keyword or tokens[j] == keyword) and \
                   word_counts[tokens[i]] >= min_occurrence and word_counts[tokens[j]] >= min_occurrence:
                    G.add_edge(tokens[i], tokens[j])

    for word in G.nodes():
        avg_sentiment = sum(word_sentiments[word]) / len(word_sentiments[word])
        G.nodes[word]['size'] = word_counts[word]
        G.nodes[word]['sentiment'] = avg_sentiment
        G.nodes[word]['icon'] = sentiment_icon(avg_sentiment)

    return G

# Filter reviews by sentiment type
def filter_reviews_by_sentiment(reviews, sentiment_type):
    if sentiment_type == "Positive":
        return reviews[reviews['sentiment'] > 0.1]
    elif sentiment_type == "Negative":
        return reviews[reviews['sentiment'] < -0.1]
    elif sentiment_type == "Neutral":
        return reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)]
    return reviews

# Page title
st.title("📊 Customer Reviews Network Graph")

# Sidebar inputs
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    excluded_words = st.text_input("Exclude Words (comma-separated)", "").lower().split(',')

    if uploaded_file:
        reviews = pd.read_csv(uploaded_file)

        # Preprocess
        reviews['tokens'] = reviews['text'].apply(lambda t: preprocess_text(t, excluded_words))
        reviews['sentiment'] = reviews['text'].apply(sentiment_analysis)

        sentiment_filter = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])
        keyword_options = ["All"] + sorted({word for tokens in reviews['tokens'] for word in tokens})
        keyword = st.selectbox("Select Keyword", keyword_options)
        node_size_scale = st.slider("Adjust Node Size", 1, 20, 10)
        min_occurrence = st.slider("Min Word Occurrence", 1, 20, 1)
        page_size = st.slider("Page Size", 5, 50, 10)

        # Filter sentiment
        reviews = filter_reviews_by_sentiment(reviews, sentiment_filter)

# Paginated reviews
if uploaded_file:
    page_number = st.number_input("Page Number", min_value=1, max_value=(len(reviews) // page_size) + 1, step=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    st.write(reviews.iloc[start_idx:end_idx])

    # Build graph
    reviews_tokens = list(zip(reviews['tokens'], reviews['sentiment']))
    G = create_network_graph(reviews_tokens, keyword, min_occurrence)
    pos = nx.spring_layout(G)

    # Edges
    edge_x, edge_y = [], []
    for e in G.edges():
        x0, y0 = pos[e[0]]
        x1, y1 = pos[e[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, mode='lines', line=dict(width=0.5, color='#888'), hoverinfo='none')

    # Nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        s = G.nodes[node]
        node_text.append(f"{node}<br>Count: {s['size']}<br>{s['icon']}")
        node_size.append(s['size'] * node_size_scale)
        node_color.append('green' if s['sentiment'] > 0.1 else 'red' if s['sentiment'] < -0.1 else 'gray')

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers+text',
        textposition="bottom center", hoverinfo='text', text=node_text,
        marker=dict(size=node_size, color=node_color, line=dict(width=2, color='#06516F'))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(title='Network graph of customer reviews',
                                    titlefont_size=16, showlegend=False, hovermode='closest',
                                    margin=dict(b=20, l=5, r=5, t=40),
                                    xaxis=dict(showgrid=False, zeroline=False),
                                    yaxis=dict(showgrid=False, zeroline=False)))
    st.plotly_chart(fig, use_container_width=True)

    # Sentiment counts
    st.subheader("📈 Sentiment Distribution")
    st.markdown(f"😊 Positive: **{(reviews['sentiment'] > 0.1).sum()}**")
    st.markdown(f"😞 Negative: **{(reviews['sentiment'] < -0.1).sum()}**")
    st.markdown(f"😐 Neutral: **{((reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)).sum()}**")
