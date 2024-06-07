# Customer Reviews Network Graph

This Streamlit app generates an interactive network graph from customer review data. The app performs text preprocessing, sentiment analysis, and visualizes the relationships between keywords in the reviews.

## How to Run the App

1. Upload a CSV file containing customer reviews with columns `id` and `text`.
2. The app will preprocess the text, perform sentiment analysis, and create a network graph.

## Features

- Search for a specific word or phrase and view the network graph centered around it.
- Zoom in and zoom out functionality for the network graph.
- Filter and view network graphs by sentiment categories (positive, negative, and neutral).
- Node size represents the occurrence of the word, adjustable for readability.
- Node color represents the sentiment (red for negative, green for positive, and gray for neutral).
- Hovering over a node shows the count and sentiment icon.
- Sentiment distribution with smileys for positive, negative, and neutral reviews.
- Pagination for the review table.
- Collapsible sidebar with all controls.

## Dependencies

- streamlit
- pandas
- networkx
- plotly
- nltk
- textblob
