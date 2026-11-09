"""
Validare: comparam implementarea proprie cu sklearn.naive_bayes.MultinomialNB
pe acelasi split train/test si aceleasi token-uri.
"""
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

from main import ALPHA, ClasificatorBayesian, load_and_split, preprocess_text, stop_words


def main():
    train_df, test_df = load_and_split()
    train_df['tokens'] = train_df['cleaned_review'].apply(preprocess_text, stop_words=stop_words)
    test_df['tokens'] = test_df['cleaned_review'].apply(preprocess_text, stop_words=stop_words)

    # Modelul propriu
    model = ClasificatorBayesian(alpha=ALPHA)
    model.fit(train_df)
    predictii_proprii = model.predict(test_df)

    # sklearn: folosim direct token-urile noastre, fara tokenizare/preprocesare proprie sklearn
    vectorizer = CountVectorizer(analyzer=lambda tokens: tokens)
    X_train = vectorizer.fit_transform(train_df['tokens'])
    X_test = vectorizer.transform(test_df['tokens'])

    nb = MultinomialNB(alpha=ALPHA)
    nb.fit(X_train, train_df['sentiment'])
    predictii_sklearn = nb.predict(X_test)

    true_labels = test_df['sentiment'].values
    acc_proprie = (predictii_proprii == true_labels).mean()
    acc_sklearn = (predictii_sklearn == true_labels).mean()
    acord = (predictii_proprii == predictii_sklearn).mean()

    print(f"Acuratete implementare proprie: {acc_proprie:.4f}")
    print(f"Acuratete sklearn MultinomialNB: {acc_sklearn:.4f}")
    print(f"Predictii identice: {acord:.2%}")


if __name__ == "__main__":
    main()
