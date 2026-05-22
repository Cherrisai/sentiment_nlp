# NLP Sentiment Analysis

Business objective: extract sentiment from ecommerce customer reviews, focused on Amazon-style product review text.

This project trains a machine learning sentiment classifier from the provided workbook `dataset -P676 (1).xlsx`, saves reusable model artifacts, and deploys the final prediction flow through Streamlit.

The Streamlit app includes single-review prediction, batch prediction, dashboard charts, sentiment filters, a review question-answer assistant, and animated sentiment result cards.

deploy: https://huggingface.co/spaces/cherrisai/sentiment_nlp?logs=container

## Project Structure

```text
.
|-- app.py
|-- data
|   |-- processed
|   `-- raw
|       |-- dataset -P676 (1).xlsx
|       |-- P676 NLP Group Lists.xlsx
|       `-- Project_Objective P676 (1).docx
|-- models
|-- notebooks
|   `-- train_file_clean_output.ipynb
|-- reports
|-- requirements.txt
|-- src
|   `-- sentiment_analysis
`-- train.py
```

## Setup

```bash
python3 -m pip install -r requirements.txt
```

## Train The Model

```bash
python3 train.py
```

Training creates:

- `models/sentiment_pipeline.joblib`
- `reports/metrics.json`
- `reports/classification_report.csv`
- `data/processed/reviews_with_sentiment.csv`

## Run Streamlit App

```bash
streamlit run app.py
```

## Labeling Rule

The dataset does not include manual sentiment labels, so labels are created from product ratings:


When an uploaded batch file contains a rating column, the app uses that rating as the strongest sentiment signal and uses review/title words plus the trained model as supporting explanation.
# sentiment_nlp
