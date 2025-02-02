# nice to have
collection.query(
    query_ids=["123", "456"],
    n_results=10,
)

# for bm_25 documents must be provided
# for hybrid, both documents and embeddings are needed
collection.query(
    query_texts=["abc", "def"],
    search_type="bm_25", # defaults to semantic, can be bm_25 or hybrid
    metadata={ # optional
        "bm_25:k1": 1.2, 
        "bm_25:b": 0.75,
        }, 
)

# reference https://www.assembled.com/blog/better-rag-results-with-reciprocal-rank-fusion-and-hybrid-search#the-solution-hybrid-search-with-reciprocal-rank-fusion
collection.query(
    query_texts=["abc", "def"],
    search_type="hybrid",
)