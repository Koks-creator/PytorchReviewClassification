import re
import string
from dataclasses import dataclass
import torch
import pandas as pd
import tensorflow
import tensorflow as tf


@dataclass
class DataCleaning:

    @staticmethod
    def remove_html_tags(raw_text: str) -> str:
        cleanr = re.compile("<.*?>")
        cleantext = re.sub(cleanr, '', raw_text)
        return cleantext

    @staticmethod
    def remove_url(text: str) -> str:
        url_pattern = re.compile(r"http[s]?://\S+.\S+.\S+")
        return url_pattern.sub(r"", text)

    @staticmethod
    def remove_punct(text: str) -> str:
        translator = str.maketrans("", "", string.punctuation)
        return text.translate(translator)

    @staticmethod
    def remove_non_ascii(text: str) -> str:
        pattern = re.compile(r"[^\x00-\x7f][^\x8d,\x10][ ]?")
        return pattern.sub(r"", text)

    @staticmethod
    def remove_digits(text: str) -> str:
        pattern = re.compile(r"\b\d+\b")
        return pattern.sub(r"", text)


class Predictor(DataCleaning):
    def __init__(self) -> None:
        super().__init__()
        self.review_col, self.label_col, self.class_name_col = ["Review", "ClassIndex", "Class"]
        self.classes = ["Positive", "Negative"]

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        data[self.review_col] = data[self.review_col].apply(lambda x: self.remove_html_tags(x))
        data[self.review_col] = data[self.review_col].apply(lambda x: self.remove_url(x))
        data[self.review_col] = data[self.review_col].apply(lambda x: self.remove_punct(x))
        data[self.review_col] = data[self.review_col].apply(lambda x: self.remove_non_ascii(x))
        data[self.review_col] = data[self.review_col].apply(lambda x: self.remove_digits(x))

        return data

    def make_predictions(self, model, sents_list: list,
                         tokenizer: tensorflow.keras.preprocessing.text.Tokenizer,
                         max_length: int = 200, device=None) -> pd.DataFrame:

        df = pd.DataFrame({self.review_col: sents_list})
        df_original = df.copy()
        cleaned_df = self.clean_data(df)

        reviews = cleaned_df[self.review_col].to_numpy()
        sequences = tokenizer.texts_to_sequences(reviews)
        sequences_padded = tf.keras.utils.pad_sequences(sequences, maxlen=max_length, padding="pre", truncating="pre")
        sequences_padded = torch.from_numpy(sequences_padded)

        if device:
            sequences_padded = sequences_padded.to(device)

        predictions = model(sequences_padded)
        predictions_rounded = torch.round(predictions).int()
        predictions_rounded = predictions_rounded.reshape(predictions_rounded.shape[0])

        df_original[self.label_col] = predictions_rounded.tolist()
        df_original[self.class_name_col] = df_original[self.label_col].apply(lambda x: self.classes[x])

        return df_original
