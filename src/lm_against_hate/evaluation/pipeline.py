import pandas as pd

from evaluation.metrics import *
from ..config.eval_config import MODEL_PATHS

def evaluation_pipeline(dfs:list, infos:dict, args:dict) -> list:
    """
    Evaluate models based on multiple metrics and return comprehensive results.
    
    Args:
        dfs: List of dataframes containing evaluation data.
        infos: Metadata about test sets and models.
        args: Dictionary of evaluation parameters.
        
    Returns:
        List of dictionaries containing evaluation results for each model.
    """
    scores = {
        "counter_argument_score": compute_argument_score(dfs, args),                                                            # 1. Counter-Argument Score
        "toxicity_score": aggregate_toxicity_scores(dfs, args, model_paths=MODEL_PATHS['toxicity']),                            # 2. Toxicity Score
        "CoLA_score": compute_cola_score(dfs, args),                                                                            # 3. CoLA Score
        "HNO_score": compute_Offense_Hate_score(dfs, args),                                                                     # 4. Hate/Normal/Offensive Score
        "repetition_rate": calculate_ngram_repetition_rate(dfs, args),                                                          # 5. Repetition Rate
        "topic_relevance_score": compute_topicRelevance_score(dfs, args, infos),                                                # 6. Topic Relevance Score
        "Context_sim_score": compute_similarity_pipeline(dfs, args, model_paths=MODEL_PATHS['context_sim'], task='context')     # 7. HS-CN Similarity Score
                                                                                  
    }
    
    # 8. CN-Label Similarity Score
    if infos[0]["Test_Set"] != "Sexism":
        scores["Label_sim_score"] = compute_similarity_pipeline(dfs, args, model_paths=MODEL_PATHS['label_sim'], task='label')
        
    eval_results = []
    # Combine scores into final results
    for df_idx, info in enumerate(infos):
        result = {**info}
        for score_name, score in scores.items():
            if isinstance(score, list):  # Per-dataset scores
                result[score_name] = score[df_idx]
            elif isinstance(score, dict):  # Multi-metric dictionaries
                result.update(score[df_idx])
            '''
            eval_["Toxicity"] = toxicity_scores[df_idx]
            eval_["CoLA"] = CoLA_scores[df_idx]
            
            eval_["Hate"] = HNO_scores[df_idx][0]
            eval_["Normal"] = HNO_scores[df_idx][1]
            eval_["Offensive"] = HNO_scores[df_idx][2]
            
            if infos[0]["Test_Set"] != "Sexism":
                eval_["Label_Cosine_Sim"] = statistics.fmean(
                    Label_sim_scores[df_idx][:])
                    
            eval_["Context_Cosine_Sim"] = statistics.fmean(
                Context_sim_scores[df_idx][:])
                
            eval_["Counter_Argument"] = CA_scores[df_idx]
            eval_["Repetition_Rate"] = 1 - repetition_rate[df_idx]
            eval_["Topic_F1"] = topics[df_idx]["f1"]
            '''
            result["G-Score"] = compute_g_score(result, infos)
            eval_results.append(result)
            
    return eval_results


def save_results(save_dir, eval):
    df = pd.DataFrame(eval)
    df.to_csv(save_dir)