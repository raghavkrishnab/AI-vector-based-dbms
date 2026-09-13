"""
sample_data.py
--------------
A small demo corpus with categories, so the app has something to search on the
very first run. Import SAMPLE_DOCS and load them into the SearchEngine.
"""

SAMPLE_DOCS = [
    # --- Sports ------------------------------------------------------------
    {"text": "I love playing basketball on weekends with my friends.", "metadata": {"category": "sports"}},
    {"text": "Football is my favorite sport to watch and play.", "metadata": {"category": "sports"}},
    {"text": "The tennis championship final went to a thrilling fifth set.", "metadata": {"category": "sports"}},
    {"text": "Our cricket team won the tournament after a nail-biting finish.", "metadata": {"category": "sports"}},

    # --- Technology --------------------------------------------------------
    {"text": "Machine learning models can recognize patterns in huge datasets.", "metadata": {"category": "technology"}},
    {"text": "Neural networks are inspired by the structure of the human brain.", "metadata": {"category": "technology"}},
    {"text": "Cloud computing lets companies scale their servers on demand.", "metadata": {"category": "technology"}},
    {"text": "A vector database stores embeddings for fast similarity search.", "metadata": {"category": "technology"}},

    # --- Food --------------------------------------------------------------
    {"text": "This restaurant serves the best wood-fired pizza in town.", "metadata": {"category": "food"}},
    {"text": "A premium canine diet keeps your dog healthy and energetic.", "metadata": {"category": "food"}},
    {"text": "Fresh basil and ripe tomatoes make the perfect pasta sauce.", "metadata": {"category": "food"}},

    # --- Royalty / classic semantic-match demo -----------------------------
    {"text": "The king is sitting on the throne in the grand hall.", "metadata": {"category": "royalty"}},
    {"text": "The monarch occupies the royal seat during the ceremony.", "metadata": {"category": "royalty"}},
    {"text": "A person was sitting on a chair by the window.", "metadata": {"category": "general"}},

    # --- Weather / distractors --------------------------------------------
    {"text": "The weather is sunny today with a gentle breeze.", "metadata": {"category": "weather"}},
    {"text": "Heavy rainfall is expected across the coast this weekend.", "metadata": {"category": "weather"}},
]

# Ready-made queries the user can click in the demo UI.
EXAMPLE_QUERIES = [
    "I enjoy sports",
    "king on throne",
    "artificial intelligence and data",
    "good food for pets",
]
