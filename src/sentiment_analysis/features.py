from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_tfidf_vectorizer() -> FeatureUnion:
    word_features = TfidfVectorizer(
        max_features=12000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.92,
        sublinear_tf=True,
        stop_words="english",
    )

    char_features = TfidfVectorizer(
        analyzer="char_wb",
        max_features=6000,
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
    )

    return FeatureUnion(
        transformer_list=[
            ("word_tfidf", word_features),
            ("char_tfidf", char_features),
        ]
    )
