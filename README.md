# Proiect: Clasificator de Sentimente Naive Bayes

Acesta este un proiect care implementează un clasificator de sentimente **Multinomial Naive Bayes** de la zero, folosind Python și biblioteca `pandas`. Scopul este de a clasifica review-uri de filme din setul de date IMDB ca fiind 'pozitive' sau 'negative'.

---

## 1. Ce face programul

Scriptul încarcă un set de date (CSV), îl curăță de cuvinte nefolositoare pentru clasificare (stop words), apoi îl împarte într-un set de antrenament (80%) și unul de test (20%).

Folosind setul de antrenament, antrenează un model bayesian naiv. Acest proces implică calcularea probabilităților ca anumite cuvinte să apară în review-uri pozitive versus negative.

Apoi, scriptul folosește modelul antrenat pentru a prezice sentimentul fiecărui review din setul de test și raportează acuratețea finală a modelului.

---

## 2. Modelul Matematic

Modelul se bazează pe **Teorema lui Bayes** cu ipoteza că toate cuvintele dintr-un document sunt independente.

Vrem să găsim categoria (sentimentul $C$) care are probabilitatea cea mai mare, dat fiind documentul (care conține cuvintele $W$):

$$P(C | W) \propto P(C) \cdot P(W | C)$$

Datorită ipotezei "naive", calculăm $P(W | C)$ ca un produs al probabilităților fiecărui cuvânt:

$$P(W | C) \approx \prod_{i=1}^{n} P(w_i | C)$$

### Probleme și Soluții

1.  Înmulțirea multor probabilități mici (0.0...1) în Python face ca rezultatul final să fie rotunjit la 0.
    * **Soluție**: Folosim logaritmi. $\log(a \cdot b) = \log(a) + \log(b)$. Căutăm scorul care maximizează:
        $$\log(P(C)) + \sum_{i=1}^{n} \log(P(w_i | C))$$

2.  Un cuvânt din test care nu a fost în setul de antrenament ar avea probabilitatea 0, anulând tot scorul.
    * **Soluție**: **Netezirea Laplace (Additive Smoothing)**. Adăugăm o constantă `ALPHA` (aici, `ALPHA = 1`) la fiecare numărător. Formula devine:
        $$P(w_i | C) = \frac{\text{count}(w_i, C) + \alpha}{N_C + V}$$
        unde $N_C$ este numărul total de cuvinte din clasa $C$ și $V$ este mărimea vocabularului (numărul total de cuvinte distincte din setul de date de antrenament).

---

## 3. Structura Codului

Scriptul este împărțit în trei părți principale:

### `preprocess_text(text, stop_words)`
* O funcție ajutătoare care primește un string.
* Îl împarte în cuvinte și elimină toate cuvintele din setul `stop_words` primit (implicit, lista globală `stop_words`).
* Returnează o listă de token-uri (cuvinte).

### `ClasificatorBayesian`
O clasă care încapsulează logica modelului.

* **`__init__(self, alpha)`**: Inițializează constantele și variabilele modelului (care vor fi calculate în `fit`).
* **`fit(self, train_df)`**: Metoda de "antrenare".
    1.  Calculează probabilitățile *a priori* ale categoriilor $P(C)$ (folosind `value_counts(normalize=True)`).
    2.  Calculează mărimea vocabularului ($V$) și numărul total de cuvinte per categorie ($N_C$) (folosind `explode`, `nunique` și `groupby`).
    3.  Calculează probabilitățile $P(w_i | C)$ pentru fiecare cuvânt și fiecare clasă, folosind formula Laplace.
    4.  Stochează aceste probabilități în `self.prob_positive` și `self.prob_negative`.
* **`predict(self, test_df)`**: Metoda de "prezicere".
    1.  Iterează prin setul de test.
    2.  Pentru fiecare review, calculează scorul logaritmic pentru 'pozitiv' și 'negativ' adunând log-probabilitățile (așa cum e descris în modelul matematic).
    3.  Alege clasa cu scorul logaritmic mai mare (cel mai apropiat de 0).
    4.  Returnează o listă de predicții.

### `load_and_split()`, `evalueaza()` și `main()`
1.  **Încarcă datele** (`load_and_split`): `pd.read_csv("IMDB_cleaned.csv")`, `df.dropna(...)`
2.  **Împarte datele** (`load_and_split`): `df.sample(frac=1, random_state=42)` și `df.iloc[...]` pentru a obține `train_df` (80%) și `test_df` (20%). Seed-ul fix face ca rezultatele să fie reproductibile.
3.  **Preprocesează** (`evalueaza`): `df.apply(preprocess_text)` cu lista de stop words primită.
4.  **Antrenează**: `model = ClasificatorBayesian()`, `model.fit(train_df)`
5.  **Prezice**: `predictii = model.predict(test_df)`
6.  **Evaluează**: Compară `predictii` cu etichetele reale din `test_df` și returnează acuratețea.

`main()` rulează `evalueaza` de două ori pe același split: o dată cu lista de stop words actuală și o dată cu lista inițială, care elimina și cuvintele de sentiment (vezi secțiunea 6).

---

## 4. Funcții Pandas Cheie

* **`.apply(func)`**: Aplică funcția `func` pe fiecare rând sau coloană. O folosim pentru a rula `preprocess_text` pe fiecare review.
* **`.value_counts()`**: Numără aparițiile unice și le returnează ca probabilități (procente) în loc de numere absolute.
* **`.explode()`**: Transformă o coloană de liste (ex: `[['a', 'b'], ['c']]`) într-o coloană lungă (ex: `['a', 'b', 'c']`), păstrând indexul original.
* **`.groupby(level=0).count()`**: După `explode()`, grupăm după index (care la noi este sentimentul) și numărăm cuvintele totale pentru fiecare sentiment.
* **`.nunique()`**: Numără elementele unice (folosit pentru a găsi mărimea vocabularului $V$).

---

## 5. Instrucțiuni de Utilizare

### Cerințe
* Python 3
* Pandas (`pip install pandas`)
* scikit-learn, doar pentru `compare_sklearn.py` (`pip install scikit-learn`)
* Fișierul `IMDB_cleaned.csv` (trebuie să conțină coloanele `cleaned_review` și `sentiment`).
* Link pentru setul de date folosit: https://www.kaggle.com/datasets/ibrahimqasimi/imdb-50k-cleaned-movie-reviews/data
* Rezultatele de mai jos sunt obținute pe un fișier `IMDB_cleaned.csv` de 5.000 de review-uri (2.532 negative, 2.468 pozitive), deci setul de test are 1.000 de review-uri.

### Rulare
1.  Asigură-te că `IMDB_cleaned.csv` este în același folder cu scriptul Python.
2.  Rulează scriptul din terminal: `python main.py`
3.  Opțional, validarea cu scikit-learn: `python compare_sklearn.py`

## 6. Rezultate

Toate rezultatele sunt pe split-ul 80/20 cu `random_state=42` (4.000 review-uri de antrenament, 1.000 de test).

### Stop words: impactul cuvintelor de sentiment

Versiunea inițială a listei de stop words elimina și cuvinte ca `good`, `great`, `love`, `best`, `bad`, `never`, `nothing`, care sunt printre cele mai puternice semnale de sentiment din review-uri. Le-am scos din listă și am comparat:

| Variantă | Acuratețe (seed 42) | Media pe 10 seed-uri (0–9) |
|---|---|---|
| Cu cuvintele de sentiment păstrate (actual) | **84,1%** | 83,5% |
| Cu cuvintele de sentiment eliminate (inițial) | 83,0% | 82,8% |

Câștigul e mic (+0,7 puncte procentuale în medie), dar consecvent: varianta actuală a fost mai bună în 9 din 10 split-uri și egală în al zecelea. Diferența e mică pentru că modelul are multe alte cuvinte cu sentiment puternic (`excellent`, `worst`, `waste` etc.), deci eliminarea câtorva nu îl afectează mult.

### Validare cu scikit-learn

`compare_sklearn.py` antrenează `sklearn.naive_bayes.MultinomialNB(alpha=1)` pe aceleași token-uri și același split:

| Model | Acuratețe |
|---|---|
| Implementarea proprie | 84,1% |
| sklearn `MultinomialNB` | 84,0% |

Predicțiile sunt identice pentru 995 din 1.000 de review-uri. Cele 5 diferențe vin din tratarea cuvintelor care nu apar deloc în setul de antrenament: sklearn le ignoră, iar implementarea proprie le atribuie probabilitatea Laplace $lpha / (N_C + V)$. Deoarece $N_C$ diferă între clase, fiecare cuvânt necunoscut împinge ușor scorul spre o clasă. Am verificat că, dacă ignorăm aceste cuvinte, predicțiile coincid 100% cu sklearn.

---

## 7. Bibliografie
* https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html