"""Run once before opening the unchanged EDA notebook."""

import nltk

if __name__ == "__main__":
    for resource in ("stopwords", "wordnet", "omw-1.4"):
        if not nltk.download(resource, raise_on_error=True):
            raise RuntimeError(f"Could not download NLTK resource: {resource}")
