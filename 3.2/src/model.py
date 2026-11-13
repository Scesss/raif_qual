"""Sparse feature channels for a linear classifier; all fitting uses train only."""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


class SelectField(TransformerMixin, BaseEstimator):
    def __init__(self, field):
        self.field = field

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [row[self.field] for row in X]


def build_model(args) -> Pipeline:
    def vectorizer(max_features, ngram_range=(1, 1)):
        return TfidfVectorizer(
            lowercase=False, token_pattern=r"(?u)\b\w+\b", min_df=args.min_df,
            max_features=max_features, sublinear_tf=True, dtype=np.float32,
            ngram_range=ngram_range,
        )

    if args.feature_set == "baseline":
        features = vectorizer(args.max_features)
    else:
        # FeatureUnion concatenates sparse matrices using scipy.sparse.hstack.
        features = FeatureUnion([
            ("bag", Pipeline([("select", SelectField("text")), ("tfidf", vectorizer(args.max_features))])),
            ("sequence", Pipeline([
                ("select", SelectField("sequence")),
                ("tfidf", vectorizer(args.sequence_max_features, (1, args.ngram_max))),
            ])),
            ("decay", Pipeline([
                ("select", SelectField("decay")),
                ("counts", DictVectorizer(dtype=np.float32)),
                # Fractional counts already contain exp(-age * ln(2) / half_life).
                # No sublinear_tf: log() would distort fractional decay weights.
                ("tfidf", TfidfTransformer(sublinear_tf=False)),
            ])),
            ("calendar", Pipeline([
                ("select", SelectField("calendar")),
                ("tfidf", vectorizer(args.max_features)),
            ])),
            ("context", Pipeline([
                ("select", SelectField("context")),
                # String-valued columns become one-hot; normalized ratios stay numeric.
                ("onehot", DictVectorizer(dtype=np.float32)),
            ])),
        ], transformer_weights={
            "bag": 1.0,
            "sequence": args.auxiliary_weight,
            "decay": args.auxiliary_weight,
            "calendar": args.auxiliary_weight,
            "context": args.context_weight,
        })
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            C=args.C, solver="lbfgs", max_iter=args.max_iter, tol=args.tol, random_state=args.seed,
        )),
    ])


def feature_sizes(model: Pipeline) -> dict[str, int]:
    features = model["features"]
    if isinstance(features, TfidfVectorizer):
        return {"bag": len(features.vocabulary_)}
    sizes = {}
    for name, channel in features.transformer_list:
        if name == "decay":
            sizes[name] = len(channel["counts"].vocabulary_)
        elif name == "context":
            sizes[name] = len(channel["onehot"].vocabulary_)
        else:
            sizes[name] = len(channel["tfidf"].vocabulary_)
    return sizes
