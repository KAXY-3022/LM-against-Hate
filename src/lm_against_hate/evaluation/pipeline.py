import pandas as pd

from lm_against_hate.evaluation.metrics import *
from lm_against_hate.config.eval_config import MODEL_PATHS


def testing(dfs: list, infos: dict, args: dict):
    topic_scores = compute_topicRelevance_score(dfs, args, infos)
    for df_idx, info in enumerate(infos):
        result = {**info}
        result.update({
            "topic_f1": float(topic_scores[df_idx]["f1"]),
            # "topic_roc_auc": float(topic_scores[df_idx]["roc_auc"]),
            "topic_accuracy": float(topic_scores[df_idx]["accuracy"])
        })
    return print(result)

def evaluation_pipeline(dfs:list, infos:dict, args:dict) -> list:
    """
    Evaluate models based on multiple metrics and return comprehensive results.
    
    Args:
        dfs: List of dataframes containing evaluation data.
        infos: Metadata about test sets and models.
        args: Dictionary of evaluation parameters including:
            - perspective_api_key: Perspective API key
            - proxy_info: Optional proxy configuration for API calls
        
    Returns:
        List of dictionaries containing evaluation results for each model.
    """
    scores = {}
    scores_per_input = {}
    
    scores["counter_argument_score"], scores_per_input["counter_argument_score"] = compute_argument_score(dfs, args)
    scores["toxicity_score"], scores_per_input["toxicity_score"] = aggregate_toxicity_scores(dfs, args, model_paths=MODEL_PATHS['toxicity'])
    scores["CoLA_score"], scores_per_input["CoLA_score"] = compute_cola_score(dfs, args)
    scores["repetition_rate"] = calculate_ngram_repetition_rate(dfs, args)
    scores["Context_sim_score"], scores_per_input["Context_sim_score"] = compute_similarity_pipeline(dfs, MODEL_PATHS['context_sim'], task='context')
    scores["Response_Length"], scores_per_input["Response_Length"] = compute_response_length(dfs)

    # 8. CN-Label Similarity Score

    if infos[0]["Test_Set"] != "Sexism":
        scores["Label_sim_score"], scores_per_input["Label_sim_score"] = compute_similarity_pipeline(
            dfs, MODEL_PATHS['label_sim'], task='label')

    # Get multi-metric scores
    hno_scores, hno_scores_per_input = compute_Offense_Hate_score(dfs, args)
    topic_scores, topic_per_input = compute_topicRelevance_score(dfs, args, infos)
    scores_per_input["topic"] = topic_per_input
    # Get Perspective API scores if API key is provided
    if "perspective_api_key" in args:
        perspective_scores = compute_perspective_api_score(
            dfs, 
            args["perspective_api_key"],
            args.get("proxy_info")
        )

    eval_results = []
    new_dfs = []
    # Combine scores into final results
    for df_idx, info in enumerate(infos):
        result = {**info}
        new_df = dfs[df_idx].copy()
        new_df["Prediction_File"] = info['Prediction_File']
        
        # Add base scores
        arrays1 = [np.array(x)
                  for x in [hno_scores_per_input[df_idx], scores_per_input["toxicity_score"][df_idx]]]
        mean_1 = [np.mean(k).item() for k in zip(*arrays1)]
        scores_per_input["toxicity_score"][df_idx] = mean_1
        
        arrays2 = [np.array(x)
                  for x in scores_per_input["Context_sim_score"][df_idx]]
        mean_2 = [np.mean(k).item() for k in zip(*arrays2)]
        scores_per_input["Context_sim_score"][df_idx] = mean_2
        
        if 'Label_sim_score' in scores_per_input:
            arrays3 = [np.array(x)
                      for x in scores_per_input["Label_sim_score"][df_idx]]
            mean_3 = [np.mean(k).item() for k in zip(*arrays3)]
            scores_per_input["Label_sim_score"][df_idx] = mean_3
            
        for score_name, score in scores.items():
            if isinstance(score, np.ndarray):
                if score.ndim == 1:
                    result[score_name] = float(score[df_idx])
                else:
                    # For multi-dimensional arrays, take mean across models
                    result[score_name] = float(np.mean(score[df_idx]))
                    
        for score_name, score in scores_per_input.items():
            if score_name == "Context_sim_score":
                print(score)
            if len(score[df_idx]) != len(new_df):
                arrays = [np.array(x) for x in score[df_idx]]
                mean_ = [np.mean(k) for k in zip(*arrays)]
                new_df[score_name] = mean_
            else:
                new_df[score_name] = score[df_idx]

        # Add HNO scores
        result.update({
            "toxicity_score": float(np.mean([hno_scores[df_idx], result["toxicity_score"]], axis=0))
            })
        

        
        # Add topic relevance scores
        result.update({
            "topic_f1": float(topic_scores[df_idx]["f1"]),
            #"topic_roc_auc": float(topic_scores[df_idx]["roc_auc"]),
            "topic_accuracy": float(topic_scores[df_idx]["accuracy"])
        })

        # Add Perspective API scores if available
        if "perspective_api_key" in args:
            for attr, score in perspective_scores[df_idx].items():
                result[f"perspective_{attr.lower()}"] = float(score)
                
                
        # Add RELEVANCE scores
        relevance_scores = [result["Context_sim_score"],
                            result["topic_f1"],
                            result["topic_accuracy"]]

        if infos[0]["Test_Set"] != "Sexism":
            relevance_scores.append(result["Label_sim_score"])
  
        result["relevance_score"] = float(np.mean(relevance_scores, axis=0))
        
        
        new_dfs.append(new_df)
        # Compute G-Score
        result["G-Score"] = compute_g_score(result, infos[0]["Test_Set"] == "Sexism")
        eval_results.append(result)
            
    return eval_results, new_dfs

def save_results(save_dir, eval):
    """
    Save evaluation results to CSV file.
    
    Args:
        save_dir (Path): Path object specifying where to save the evaluation results CSV file.
            Will create parent directories if they don't exist.
        eval (List[Dict]): List of dictionaries containing evaluation results.
            Each dictionary contains:
                - Model_Name (str): Name of the model used for generation
                - Model_Version (str): Version/timestamp of the model
                - Test_Set (str): Name of test dataset used
                - Prediction_File (str): Name of prediction file evaluated
                - counter_argument_score (float): Quality of counter-arguments (0-1)
                - toxicity_score (float): Level of toxicity (0-1)
                - CoLA_score (float): Grammatical acceptability (0-1)
                - repetition_rate (float): N-gram repetition rate (0-1)
                - Context_sim_score (float): Semantic similarity to context (0-1)
                - Label_sim_score (float): Semantic similarity to ground truth (0-1)
                - hate (float): Hate speech detection score (0-1)
                - normal (float): Normal speech detection score (0-1)
                - offensive (float): Offensive speech detection score (0-1)
                - topic_f1 (float): Topic relevance F1 score (0-1)
                - topic_roc_auc (float): Topic ROC AUC score (0-1)
                - topic_accuracy (float): Topic classification accuracy (0-1)
                - G-Score (float): Overall geometric mean quality score (0-1)
            
    Returns:
        None. Saves results to CSV file at specified path with scores rounded to 4 decimal places.
    """
    # Create parent directories if they don't exist
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    
    # Round scores to 4 decimal places before saving
    for result in eval:
        for key, value in result.items():
            if isinstance(value, float):
                result[key] = round(value, 4)
    
    df = pd.DataFrame(eval)
    df.to_csv(save_dir)