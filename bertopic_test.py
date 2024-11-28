import os
import pickle
import textwrap

from adjustText import adjust_text
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
import itertools
from pathlib import Path

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance, TextGeneration
from hdbscan import HDBSCAN
from umap import UMAP

import transformers
from sentence_transformers import SentenceTransformer

from torch import bfloat16, cuda
import pandas as pd

from param import model_path


def main():
    # set data paths
    data_dir = Path().resolve().joinpath('predictions', 'Sexism')
    model_dir = model_path.joinpath('Causal', 'Llama-3.2-3B-Instruct')

    ori_dir = data_dir.joinpath(
        'Llama-3.2-1B-Instruct_16,11,2024--00,22_Sexism_26,11,2024--20,35.csv')
    cat_dir = data_dir.joinpath(
        'Llama-3.2-1B-Instruct-category_17,11,2024--09,39_Sexism_26,11,2024--20,57.csv')

    print('loading predictions')
    df_ori = pd.read_csv(ori_dir).fillna('')
    df_cat = pd.read_csv(cat_dir).fillna('')

    hate_speech = df_ori['Hate_Speech']
    ori_counter = df_ori['Prediction']
    cat_counter = df_cat['Prediction']

    docs = ori_counter

    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

    print(device)

    # set quantization configuration to load large model with less GPU memory
    # this requires the `bitsandbytes` library

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # 4-bit quantization
        bnb_4bit_quant_type='nf4',  # Normalized float 4
        bnb_4bit_use_double_quant=True,  # Second quantization after the first
        bnb_4bit_compute_dtype=bfloat16  # Computation type
    )

    print('loading generator')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map='auto',
    )
    model.eval()

    # Our text generator
    generator = transformers.pipeline(
        model=model, tokenizer=tokenizer,
        task='text-generation',
        temperature=0.1,
        max_new_tokens=500,
        repetition_penalty=1.1
    )

    print('creating topic generation prompt')
    # System prompt describes information given to all conversations
    system_prompt = """
    <s>[INST] <<SYS>>
    You are a helpful, respectful and honest assistant for labeling topics.
    <</SYS>>
    """
    # Example prompt demonstrating the output we are looking for
    example_prompt = """
    I have a topic that contains the following documents:
    - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
    - Meat, but especially beef, is the word food in terms of emissions.
    - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

    The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

    [/INST] Environmental impacts of eating meat
    """
    # Our main prompt with documents ([DOCUMENTS]) and keywords ([KEYWORDS]) tags
    main_prompt = """
    [INST]
    I have a topic that contains the following documents:
    [DOCUMENTS]

    The topic is described by the following keywords: '[KEYWORDS]'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
    [/INST]
    """

    prompt = system_prompt + example_prompt + main_prompt

    print('loading embedding model')
    embedding_model = SentenceTransformer(str(model_path.joinpath(
        'Classifiers', 'sentence-transformers', 'stella_en_400M_v5')),trust_remote_code=True).cuda()

    print('encoding predictions')
    embeddings = embedding_model.encode(
        docs, prompt_name="s2s_query", batch_size=128, show_progress_bar=True)

    print('loading umap model')
    umap_model = UMAP(n_neighbors=15, n_components=5,
                      min_dist=0.0, metric='cosine', random_state=42)

    print('loading hdbscan model')
    hdbscan_model = HDBSCAN(min_cluster_size=40, metric='euclidean',
                            cluster_selection_method='eom', prediction_data=True)

    # Pre-reduce embeddings for visualization purposes
    print('dimensionality reduction running')
    reduced_embeddings = UMAP(n_neighbors=15, n_components=2, min_dist=0.0,
                              metric='cosine', random_state=42).fit_transform(embeddings)

    print('initializing labeling models')
    # KeyBERT
    keybert = KeyBERTInspired()

    # MMR
    mmr = MaximalMarginalRelevance(diversity=0.3)

    # Text generation with Llama 2
    llama3 = TextGeneration(generator, prompt=prompt)

    # All representation models
    representation_model = {
        "KeyBERT": keybert,
        "Llama3": llama3,
        "MMR": mmr,
    }

    print('loading topic model')
    topic_model = BERTopic(

        # Sub-models
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        representation_model=representation_model,

        # Hyperparameters
        top_n_words=10,
        verbose=True
    )

    print('training starts')
    # Train model
    topics, probs = topic_model.fit_transform(docs, embeddings)

    print('visualizing data distribution')
    # Show topics
    topic_model.get_topic_info()

    topic_model.get_topic(1, full=True)["KeyBERT"]

    llama3_labels = [label[0][0].split(
        "\n")[0] for label in topic_model.get_topics(full=True)["Llama3"].values()]
    topic_model.set_topic_labels(llama3_labels)

    fig = topic_model.visualize_documents(docs,
                                          reduced_embeddings=reduced_embeddings,
                                          hide_annotations=True,
                                          hide_document_hover=False,
                                          custom_labels=True,
                                          width=1200,
                                          height=1200)

    fig.show()

    fig2 = topic_model.visualize_document_datamap(docs,
                                                  reduced_embeddings=reduced_embeddings,
                                                  custom_labels=True,
                                                  width=1200,
                                                  height=1200)
    
    fig2.show()
    '''
    ADVANCED VISUALIZATION
    '''
    '''# Define colors for the visualization to iterate over
    colors = itertools.cycle(['#e6194b', '#3cb44b', '#ffe119', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe',
                              '#008080', '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080', '#ffffff', '#000000'])
    color_key = {str(topic): next(colors)
                 for topic in set(topic_model.topics_) if topic != -1}

    # Prepare dataframe and ignore outliers
    df = pd.DataFrame({"x": reduced_embeddings[:, 0], "y": reduced_embeddings[:, 1], "Topic": [
        str(t) for t in topic_model.topics_]})
    df["Length"] = [len(doc) for doc in hate_speech]
    df = df.loc[df.Topic != "-1"]
    df = df.loc[(df.y > -10) & (df.y < 10) & (df.x < 10) & (df.x > -10), :]
    df["Topic"] = df["Topic"].astype("category")

    # Get centroids of clusters
    mean_df = df.groupby("Topic").mean().reset_index()
    mean_df.Topic = mean_df.Topic.astype(int)
    mean_df = mean_df.sort_values("Topic")

    fig = plt.figure(figsize=(10, 10))
    sns.scatterplot(data=df, x='x', y='y', c=df['Topic'].map(
        color_key), alpha=0.4, sizes=(0.4, 10), size="Length")

    # Annotate top 50 topics
    texts, xs, ys = [], [], []
    for row in mean_df.iterrows():
    topic = row[1]["Topic"]
    name = textwrap.fill(topic_model.custom_labels_[int(topic)], 20)

    if int(topic) <= 50:
        xs.append(row[1]["x"])
        ys.append(row[1]["y"])
        texts.append(plt.text(row[1]["x"], row[1]["y"], name, size=10, ha="center", color=color_key[str(int(topic))],
                              path_effects=[pe.withStroke(
                                  linewidth=0.5, foreground="black")]
                              ))

    # Adjust annotations such that they do not overlap
    adjust_text(texts, x=xs, y=ys, time_lim=1, force_text=(0.01, 0.02),
                force_static=(0.01, 0.02), force_pull=(0.5, 0.5))
    plt.axis('off')
    plt.legend('', frameon=False)
    plt.show()

    '''

    '''
    SAVE MODELS
    '''

    '''

    with open('rep_docs.pickle', 'wb') as handle:
        pickle.dump(topic_model.representative_docs_,
                    handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('reduced_embeddings.pickle', 'wb') as handle:
        pickle.dump(reduced_embeddings, handle,
                    protocol=pickle.HIGHEST_PROTOCOL)

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)

    embedding_model = "BAAI/bge-small-en"
    topic_model.save("final", serialization="safetensors",
                     save_ctfidf=True, save_embedding_model=embedding_model)'''


if __name__ == "__main__":
    main()
