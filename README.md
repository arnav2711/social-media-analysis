# Social Media Propaganda Analysis

Propaganda detection on news articles using the QProp (Proppy 1.0) corpus. The repo has a preprocessing and feature-analysis pipeline plus several classical ML baselines, some implemented from scratch in NumPy.

## Dataset

[Proppy Corpus 1.0 / QProp](https://zenodo.org/records/3271522): about 52k news articles from 100+ outlets, labeled propagandistic or non-propagandistic. Labels come from distant supervision: an article counts as propagandistic if its source was labeled propagandistic by human annotators (MBFC).

The `.tsv` files are gitignored because of their size, so download them from Zenodo and place them in `qprop_data/` (or `3271522/` for `tfidf_knn.py`):


```
proppy_1.0.train.tsv
proppy_1.0.dev.tsv
proppy_1.0.test.tsv
```

In every script, column 0 is `article_text` and column 14 is `propaganda_label` (`-1` maps to 0, non-propaganda; anything else maps to 1, propaganda).

## Models

| Script | Approach |
|---|---|
| `knn_classifier.py` | Toy KNN on a 4-point dataset, used to demonstrate the algorithm and plot neighbors |
| `KNN.py` / `knn_final.py` | KNN from scratch (Euclidean distance, NumPy) on 2,000 bag-of-words features, evaluated for K = 1, 3, 5, 10 on an 80/20 split of the training file. The two files are identical. |
| `tfidf_knn.py` | Same KNN, using 500 TF-IDF features (unigrams and bigrams). Reports accuracy, precision, recall and F1 for K = 1, 3, 5, 10 and plots them. |
| `bag_of_words_preprocess_and_logistic_regression.py` | Logistic regression implemented from scratch (gradient descent, learning rate 0.5, 100 epochs) on 2,000 stemmed bag-of-words features. Trains on the train file, evaluates on the test file, and prints the 10 strongest words for each class. |
| `svm_work_branch.py` | Scikit-learn `LinearSVC` on the same features and preprocessing, with the same evaluation and top-word printout |

Text cleaning: lowercase, strip punctuation, remove stopwords. The logistic regression and SVM scripts also remove digits and apply Porter stemming.

## Preprocessing and Feature Analysis

`preprocess-qprop.py` runs the full analysis pipeline on `proppy_1.0.train.tsv`:

- Cleaning, text statistics, and per-source and temporal analysis
- Social-media-style features: hashtag, mention, URL, exclamation, question, ALL-CAPS and punctuation counts, plus average GDELT tone
- 5,000 character trigram features (5,013 features in total)
- Visualizations and statistics export

Committed outputs:

| File | Contents |
|---|---|
| `qprop_statistics_enhanced.json` | Summary, propaganda-pattern and temporal statistics |
| `qprop_analysis_enhanced.png` | Dataset visualizations |
| `char_trigram_vocab.json` | Character trigram vocabulary |
| `feature_names.json` | Feature documentation |

The script also writes cleaned CSVs and `.npy` feature matrices, which are gitignored.

## Setup

```bash
git clone https://github.com/arnav2711/social-media-analysis.git
cd social-media-analysis
pip install numpy pandas scikit-learn matplotlib seaborn nltk
```

NLTK stopwords are downloaded automatically on first run.

## Usage

Run scripts from the repo root, with the dataset files in place:

```bash
python preprocess-qprop.py        # needs proppy_1.0.train.tsv in the working directory
python tfidf_knn.py
python bag_of_words_preprocess_and_logistic_regression.py
python svm_work_branch.py
```

The from-scratch KNN is slow (it compares each test row against every training row), so expect long runtimes on the full dataset.

## Repository Structure

```
preprocess-qprop.py        Preprocessing, feature extraction, analysis
KNN.py, knn_final.py       KNN from scratch (bag-of-words)
tfidf_knn.py               KNN from scratch (TF-IDF)
knn_classifier.py          KNN toy demo
bag_of_words_...py         Logistic regression from scratch
svm_work_branch.py         Linear SVM
qprop_data/, 3271522/      Dataset location (data files gitignored)
```

