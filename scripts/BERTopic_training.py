from sklearn.preprocessing import LabelEncoder
from bertopic import BERTopic
from pathlib import Path
import pandas as pd
import numpy
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.dimensionality import BaseDimensionalityReduction
from sklearn.linear_model import LogisticRegression

data_path = Path().resolve().joinpath('data', 'Custom_New')
data_train = pd.read_csv(data_path.joinpath(
    'Generator_train.csv'), converters={'Target': pd.eval})
data_val = pd.read_csv(data_path.joinpath(
    'Generator_val.csv'), converters={'Target': pd.eval})
data_test = pd.read_csv(data_path.joinpath(
    'Generator_test.csv'), converters={'Target': pd.eval})

save_path = Path().resolve().joinpath('models', 'BERTopic_model')

df = pd.concat(
    [data_train, data_val, data_test],
    ignore_index=False,
    sort=True)

hate_docs = df["Hate_Speech"]
counter_docs = df["Counter_Speech"]

test_data = pd.read_csv(Path().resolve().joinpath(
    'predictions', 'Base/Human_NAN_Base_NAN.csv'))
test_data['Target'] = test_data['Target'].where((test_data['Target']!='Islamophobia'), 'MUSLIMS')
predictions = test_data['Prediction']
labels = test_data['Target']

docs = pd.concat([hate_docs, counter_docs])

targets = df["Target"]
def transform_row(r):
    if len(r) > 0:
        return r[0]
    else:
        return 'UNK'

targets = targets.apply(transform_row)

# Initialize LabelEncoder
le = LabelEncoder()
# Fit and transform the target variable
targets_encoded = le.fit_transform(labels)

# Skip over dimensionality reduction, replace cluster model with classifier,
# and reduce frequent words while we are at it.
empty_dimensionality_model = BaseDimensionalityReduction()
clf = LogisticRegression()
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

# Create a fully supervised BERTopic instance
topic_model = BERTopic(
    umap_model=empty_dimensionality_model,
    hdbscan_model=clf,
    ctfidf_model=ctfidf_model
)


topics, probs = topic_model.fit_transform(predictions, y=targets_encoded)


# Map input `y` to topics
mappings = topic_model.topic_mapper_.get_mappings()
mappings = {value: le.classes_[key]
            for key, value in mappings.items()}

topic_model.save(save_path, serialization="safetensors")
numpy.save(save_path.joinpath('classes.npy'), le.classes_)
