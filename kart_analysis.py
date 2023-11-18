from matplotlib import pyplot as plt
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def plot_lap_times(heat: pd.DataFrame):
    for name,g in heat.groupby('Driver'):
        plt.plot(g["Lap"], g['Time'].apply(lambda x: x.total_seconds()), label=name)
    plt.grid(True)
    plt.locator_params(axis="x", integer=True)
    plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
    plt.title(heat.iloc[0]["Heat"])
    plt.show()

def encode_as_categorical(laps: pd.DataFrame, columns: list[str]):
    feature_encoder = OneHotEncoder()
    features = feature_encoder.fit_transform(laps[columns]).toarray()
    feature_labels = []
    for labels in feature_encoder.categories_:
        for cat in labels:
            feature_labels.append(cat)

    features_pd = pd.DataFrame(features, columns=feature_labels)
    dataset = pd.concat([laps, features_pd], axis=1)
    return feature_encoder, features_pd, dataset

def get_model_coefficients(model, features):
    return pd.DataFrame(model.coef_, columns=features.columns).T

def plot_coefficients(coefs, feature_encoder : OneHotEncoder, category_labels: list[str]):
    categories = feature_encoder.categories_
    if len(categories) != len(category_labels):
        raise ValueError(f"Number of categoeis does not match number of labels {len(categories)} vs {len(category_labels)}")

    for idx in range(0, len(category_labels)):
        label = category_labels[idx]
        if idx == 0:
            feature_coefs = coefs.iloc[0:len(feature_encoder.categories_[idx])]
        elif idx + 1 < len(category_labels):
            startIdx = sum(map(lambda c: len(c), feature_encoder.categories_[0:idx]))
            endIdx = startIdx + len(feature_encoder.categories_[idx + 1])
            feature_coefs = coefs.iloc[startIdx:endIdx]
        else:
            startIdx = sum(map(lambda c: len(c), feature_encoder.categories_[0:idx]))
            feature_coefs = coefs.iloc[startIdx:]
        feature_coefs = feature_coefs.sort_values(ascending=False, by=0)
        feature_coefs.plot.barh(figsize=(9, 7))
        plt.title(f"Linear Model {label}")
        plt.axvline(x=0, color=".5")
        plt.xlabel("Raw coefficient values")
        plt.subplots_adjust(left=0.3)
        plt.show()

    driver_coefs = coefs.iloc[len(feature_encoder.categories_[0]):]
    driver_coefs = driver_coefs.sort_values(ascending=False, by=0)
    