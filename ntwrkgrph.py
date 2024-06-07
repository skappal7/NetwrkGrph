import streamlit as st
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from textblob import TextBlob
import nltk

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Function to preprocess text
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english')).union(set(['and', 'or', 'but', 'if', 'because']))
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Function to perform sentiment analysis
def sentiment_analysis(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

# Function to determine sentiment icon
def sentiment_icon(sentiment):
    if sentiment > 0.1:
        return "😊"
    elif sentiment < -0.1:
        return "😞"
    else:
        return "😐"

# Function to create a network graph
def create_network_graph(reviews_tokens, keyword=None):
    G = nx.Graph()
    word_counts = {}
    word_sentiments = {}

    for tokens, sentiment in reviews_tokens:
        for word in tokens:
            if word not in word_counts:
                word_counts[word] = 0
                word_sentiments[word] = []
            word_counts[word] += 1
            word_sentiments[word].append(sentiment)

    for tokens in reviews_tokens:
        for i in range(len(tokens[0])):
            for j in range(i + 1, len(tokens[0])):
                if keyword is None or (tokens[0][i] == keyword or tokens[0][j] == keyword):
                    G.add_edge(tokens[0][i], tokens[0][j])

    for word in word_counts:
        if word in G:
            avg_sentiment = sum(word_sentiments[word]) / len(word_sentiments[word])
            G.nodes[word]['size'] = word_counts[word]
            G.nodes[word]['sentiment'] = avg_sentiment
            G.nodes[word]['icon'] = sentiment_icon(avg_sentiment)

    return G

# Function to filter reviews by sentiment
def filter_reviews_by_sentiment(reviews, sentiment):
    if sentiment == "Positive":
        return reviews[reviews['sentiment'] > 0.1]
    elif sentiment == "Negative":
        return reviews[reviews['sentiment'] < -0.1]
    else:
        return reviews[(reviews['sentiment'] >= -0.1) & (reviews['sentiment'] <= 0.1)]

# Streamlit UI
st.title("Customer Reviews Network Graph")
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    reviews = pd.read_csv(uploaded_file)
    st.write("Uploaded Reviews:")
    st.write(reviews)

    # Preprocess reviews
    reviews['tokens'] = reviews['text'].apply(preprocess_text)
    reviews['sentiment'] = reviews['text'].apply(sentiment_analysis)

    sentiment_filter = st.selectbox("Select Sentiment", ["All", "Positive", "Negative", "Neutral"])
    keyword = st.selectbox("Select Keyword", sorted(set(word for tokens in reviews['tokens'] for word in tokens if word.isalpha())))

    # Slider for node size scaling
    node_size_scale = st.slider("Adjust Node Size", min_value=1, max_value=20, value=10)

    # Filter reviews by sentiment
    if sentiment_filter != "All":
        reviews = filter_reviews_by_sentiment(reviews, sentiment_filter)

    # Create network graph
    reviews_tokens = list(zip(reviews['tokens'], reviews['sentiment']))
    G = create_network_graph(reviews_tokens, keyword)
    pos = nx.spring_layout(G)

    # Create Plotly figure
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(f"{node}<br>Count: {G.nodes[node]['size']}<br>{G.nodes[node]['icon']}")
        node_size.append(G.nodes[node]['size'] * node_size_scale)
        sentiment = G.nodes[node]['sentiment']
        if sentiment > 0.1:
            node_color.append('green')
        elif sentiment < -0.1:
            node_color.append('red')
        else:
            node_color.append('gray')

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=[node for node in G.nodes()],
        textposition="bottom center",
        hoverinfo='text+text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        title='Network graph of customer reviews',
                        titlefont_size=16,
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False),
                        yaxis=dict(showgrid=False, zeroline=False))
                    )

    fig.update_layout(
        dragmode='zoom',  # Enable zoom
        clickmode='event+select'
    )

    st.plotly_chart(fig)
