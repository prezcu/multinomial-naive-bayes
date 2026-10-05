import pandas as pd
import math


ALPHA = 1
RANDOM_SEED = 42

stop_words = {'other', 'he', 'she', 'is', 'when', 'there', 'was', 'before', 'too', 'do', 'what', 'or', 'after'
    , 'we', 'them', 'his', 'their', 'had', 'will', 'have', 'does', 'theirs', 'being', 'did', 'the', 's', 'it'
    , 'yours', 'those', 'shan', 'd', 'at', 'me', 'i', 't', 'of', 'they', 'would', 'and', 'you', 'o'
    , 'are', 'by', 'our', 'here', 'own', 'as', 'ours', 'while', 'where', 'from', 'ourselves', 'himself'
    , 'were', 'now', 'between', 'him', 'than', 'yourself', 'in', 'your', 'that', 'this'
    , 'which', 'for', 'itself', 'having', 'herself', 'to', 'with', 'an', 'its', 'been', 'on', 'who'
    , 'each', 'once', 'then', 'has', 'my', 'some', 'how', 'a', 'am', 'but', 'be', 'if', 'out', 'her'
    , 'up', 'because', 'could', 'get', 'any', 'also'
    , 'film', 'movie', 'off', 'films'}

# Cuvinte cu sentiment puternic. Versiunea initiala le elimina ca stop words;
# le pastram separat ca sa putem compara acuratetea cu si fara ele.
sentiment_words = {'good', 'great', 'love', 'best', 'bad', 'never', 'nothing'}


def preprocess_text(text, stop_words=stop_words):
    """Eliminam stopwords din text"""
    if not isinstance(text, str):
        return []

    tokens = text.split()
    cleaned_tokens = []
    for word in tokens:
        if word not in stop_words:
            cleaned_tokens.append(word)
    return cleaned_tokens


class ClasificatorBayesian:

    def __init__(self, alpha=ALPHA):
        self.ALPHA = alpha

        self.class_priors = None
        self.vocabular = 0
        self.total_words = None

        self.prob_positive = {}
        self.laplace_positive = 0
        self.prob_negative = {}
        self.laplace_negative = 0

    def fit(self, train_df):
        """
        Antrenam modelul cu setul de date de antrenament

        Parametrii:
            train_df (pd.DataFrame): DataFrame cu coloanele 'sentiment' (pozitiv/negativ) si token-urile (cuvintele).
        """

        # Calculam probabilitatile ca un document sa apartina unei categorii
        self.class_priors = train_df.value_counts(subset="sentiment", normalize=True)

        # Calculam marimea vobabularului si nr total de cuvinte/categorie
        indexed_train_df = train_df.set_index('sentiment')
        all_words_series = indexed_train_df['tokens'].explode()

        self.vocabular = all_words_series.nunique()
        self.total_words = all_words_series.groupby(level=0).count()

        # Calculam probabilitatile cuvintelor din review-uri pozitive.
        words_positive = train_df.loc[train_df['sentiment'] == "positive"]['tokens']
        word_count_positive = words_positive.explode().value_counts()

        self.prob_positive = {}  # Resetam
        self.laplace_positive = self.ALPHA / (self.total_words['positive'] + self.vocabular)

        for label, value in word_count_positive.items():
            self.prob_positive[label] = (value + self.ALPHA) / (self.total_words['positive'] + self.vocabular)

        # Calculam probabilitatile cuvintelor din review-uri negative.
        words_negative = train_df.loc[train_df['sentiment'] == "negative"]['tokens']
        word_count_negative = words_negative.explode().value_counts()

        self.prob_negative = {}  # Resetam
        self.laplace_negative = self.ALPHA / (self.total_words['negative'] + self.vocabular)

        for label, value in word_count_negative.items():
            self.prob_negative[label] = (value + self.ALPHA) / (self.total_words['negative'] + self.vocabular)

        print("Fit complet.")

    def predict(self, test_df):
        """
        Prezice sentimentul pentru un review (pozitiv/negativ).

        Parametrii:
            test_df (pd.DataFrame): DataFrame cu o coloana cu token-urile review-urilor din setul de test.

        Returneaza:
            list: O lista cu predictiile sentimentelor.
        """
        print("Modelul prezice setul de date de test...")
        predictii = []

        for review in test_df.itertuples():
            # Initializam scorurile cu probabilitatile categoriilor.
            scor_pos = math.log(self.class_priors['positive'])
            scor_neg = math.log(self.class_priors['negative'])

            for word in review.tokens:
                # Scorul pozitiv
                p_pos = self.prob_positive.get(word)
                if p_pos is not None:
                    scor_pos += math.log(p_pos)
                else:
                    # Cuvantul nu a fost gasit, folosim Laplace.
                    scor_pos += math.log(self.laplace_positive)

                # Scorul negativ
                p_neg = self.prob_negative.get(word)
                if p_neg is not None:
                    scor_neg += math.log(p_neg)
                else:
                    # Cuvantul nu a fost gasit, folosim Laplace.
                    scor_neg += math.log(self.laplace_negative)

            # Alegem clasa cu log-probabilitatea mai mare (mai aproape de 0).
            rezultat = 'positive' if scor_pos > scor_neg else 'negative'
            predictii.append(rezultat)

        return predictii



def load_and_split(path="IMDB_cleaned.csv", seed=RANDOM_SEED):
    """Incarca datele si le imparte 80/20 in train/test, reproductibil prin seed."""
    df = pd.read_csv(path, usecols=['cleaned_review', 'sentiment'])
    df = df.dropna(subset=['cleaned_review'])

    df_shuffled = df.sample(frac=1, random_state=seed)
    training_size = int(len(df_shuffled) * 0.8)
    train_df = df_shuffled.iloc[:training_size].copy()
    test_df = df_shuffled.iloc[training_size:].copy()
    return train_df, test_df


def evalueaza(train_df, test_df, stop_words):
    """Tokenizeaza cu lista de stop words data, antreneaza modelul si returneaza acuratetea."""
    train_df['tokens'] = train_df['cleaned_review'].apply(preprocess_text, stop_words=stop_words)
    test_df['tokens'] = test_df['cleaned_review'].apply(preprocess_text, stop_words=stop_words)

    # Initializare si fit model
    model = ClasificatorBayesian(alpha=ALPHA)
    model.fit(train_df)

    # Predictie
    predictii = model.predict(test_df)

    # Evaluare
    true_labels = test_df['sentiment']
    total_corecte = (pd.Series(predictii) == true_labels.values).sum()
    total_test = len(test_df)

    print(f"Total corecte:{total_corecte}")
    print(f"Total teste:{total_test}")
    return total_corecte / total_test


def main():

    try:
        train_df, test_df = load_and_split()
    except FileNotFoundError:
        print("Eroare: IMDB_cleaned.csv nu a fost gasit.")
        return

    if len(test_df) == 0:
        print("Nu exista date care sa fie evaluate.")
        return

    print("--- Stop words fara cuvinte de sentiment ---")
    acc = evalueaza(train_df, test_df, stop_words)
    print(f"Corectitudine model:{acc:.4f}\n")

    print("--- Stop words cu cuvinte de sentiment eliminate (versiunea initiala) ---")
    acc_initial = evalueaza(train_df, test_df, stop_words | sentiment_words)
    print(f"Corectitudine model:{acc_initial:.4f}")


if __name__ == "__main__":
    main()
