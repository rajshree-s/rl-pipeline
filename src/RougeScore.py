from typing import List, Dict
from rouge_score import rouge_scorer


def compare_slm_rouge_scores(
        references: List[str],
        finetuned_slm_outputs: List[str],
        old_slm_outputs: List[str],
        metrics: List[str] = ['rouge1', 'rouge2']
) -> Dict[str, Dict[str, float]]:
    if not (len(references) == len(finetuned_slm_outputs) == len(old_slm_outputs)):
        raise ValueError("All input lists (references, fine-tuned outputs, old outputs) must have the same length.")

    # Initialize the scorer
    scorer = rouge_scorer.RougeScorer(metrics, use_stemmer=True)

    # Initialize score accumulators
    finetuned_scores_sum = {m: 0.0 for m in metrics}
    old_scores_sum = {m: 0.0 for m in metrics}

    num_samples = len(references)

    for ref, ft_out, old_out in zip(references, finetuned_slm_outputs, old_slm_outputs):
        # 1. Score the Fine-Tuned Model
        ft_scores = scorer.score(ref, ft_out)
        for m in metrics:
            # We typically use the F-measure (fscore) for reporting ROUGE
            finetuned_scores_sum[m] += ft_scores[m].fmeasure

        # 2. Score the Old Model
        old_scores = scorer.score(ref, old_out)
        for m in metrics:
            old_scores_sum[m] += old_scores[m].fmeasure

    # 3. Calculate Averages
    avg_finetuned_scores = {
        m: total / num_samples for m, total in finetuned_scores_sum.items()
    }
    avg_old_scores = {
        m: total / num_samples for m, total in old_scores_sum.items()
    }

    return {
        "fine_tuned_slm_avg_rouge": avg_finetuned_scores,
        "old_slm_avg_rouge": avg_old_scores,
    }