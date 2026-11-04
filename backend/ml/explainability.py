"""
Explainability and Ambiguity Detection Module for Topic Classification.
Computes token-level feature attributions and explains classification ambiguity.

Note: Feature contributions represent statistical associations (correlations) learned from
the training corpus, not causal relationships.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class ExplainabilityService:
    @staticmethod
    def explain_prediction(pipeline, headline: str, predicted_category: str, probabilities: Dict[str, float]) -> Dict[str, Any]:
        """
        Extract the most influential tokens for the prediction using feature weights.
        Supports LogisticRegression, LinearSVC, and MultinomialNB inside a Pipeline.
        """
        tfidf = pipeline.named_steps.get("tfidf")
        clf = pipeline.named_steps.get("clf")

        if tfidf is None or clf is None:
            return {
                "top_contributing_tokens": [],
                "opposing_tokens": [],
                "ambiguity_analysis": {
                    "is_ambiguous": False,
                    "confidence_margin": 1.0,
                    "competing_category": None,
                    "explanation": "Pipeline components not accessible for explanation."
                },
                "interpretation_disclaimer": "Feature contributions reflect statistical corpus correlations (TF-IDF × class weights), not causal inferences."
            }

        # Vectorize single headline
        X_vec = tfidf.transform([headline])
        feature_names = np.array(tfidf.get_feature_names_out())
        classes = list(clf.classes_)

        try:
            cat_idx = classes.index(predicted_category)
        except ValueError:
            return {
                "top_contributing_tokens": [],
                "opposing_tokens": [],
                "ambiguity_analysis": {
                    "is_ambiguous": False,
                    "confidence_margin": 1.0,
                    "competing_category": None,
                    "explanation": "Predicted category index not found."
                },
                "interpretation_disclaimer": "Feature contributions reflect statistical corpus correlations (TF-IDF × class weights), not causal inferences."
            }

        # Non-zero indices in input
        nonzero_indices = X_vec.nonzero()[1]
        if len(nonzero_indices) == 0:
            return {
                "top_contributing_tokens": [],
                "opposing_tokens": [],
                "ambiguity_analysis": {
                    "is_ambiguous": False,
                    "confidence_margin": 1.0,
                    "competing_category": None,
                    "explanation": "No known vocabulary terms detected in input headline."
                },
                "interpretation_disclaimer": "Feature contributions reflect statistical corpus correlations (TF-IDF × class weights), not causal inferences."
            }

        word_contributions = []

        # Extract fitted coefficients from LinearSVC / CalibratedClassifierCV
        fold_coefs = []
        if hasattr(clf, "calibrated_classifiers_") and len(clf.calibrated_classifiers_) > 0:
            for cal in clf.calibrated_classifiers_:
                est = getattr(cal, "estimator", getattr(cal, "base_estimator", None))
                if est is not None and hasattr(est, "coef_"):
                    fold_coefs.append(est.coef_)

        if fold_coefs:
            # CalibratedClassifierCV: Average fitted coefficients across CV folds
            avg_coef = np.mean(fold_coefs, axis=0)
            weights = avg_coef[cat_idx]
            for idx in nonzero_indices:
                token = feature_names[idx]
                tf_val = float(X_vec[0, idx])
                weight = float(weights[idx])
                contribution = float(tf_val * weight)
                word_contributions.append({
                    "token": token,
                    "tfidf_value": round(tf_val, 4),
                    "weight": round(weight, 4),
                    "contribution_score": round(contribution, 4),
                })
        elif hasattr(clf, "coef_"):
            # Direct linear model (LogisticRegression, uncalibrated LinearSVC)
            weights = clf.coef_[cat_idx]
            for idx in nonzero_indices:
                token = feature_names[idx]
                tf_val = float(X_vec[0, idx])
                weight = float(weights[idx])
                contribution = float(tf_val * weight)
                word_contributions.append({
                    "token": token,
                    "tfidf_value": round(tf_val, 4),
                    "weight": round(weight, 4),
                    "contribution_score": round(contribution, 4),
                })
        elif hasattr(clf, "feature_log_prob_"):
            # Naive Bayes model: relative log likelihood
            cat_log_prob = clf.feature_log_prob_[cat_idx]
            mean_other_log_prob = np.mean(
                [clf.feature_log_prob_[i] for i in range(len(classes)) if i != cat_idx], axis=0
            )
            for idx in nonzero_indices:
                token = feature_names[idx]
                tf_val = float(X_vec[0, idx])
                diff = float(cat_log_prob[idx] - mean_other_log_prob[idx])
                contribution = float(tf_val * diff)
                word_contributions.append({
                    "token": token,
                    "tfidf_value": round(tf_val, 4),
                    "weight": round(diff, 4),
                    "contribution_score": round(contribution, 4),
                })
        else:
            # Fallback only when model coefficients cannot be extracted
            for idx in nonzero_indices:
                token = feature_names[idx]
                tf_val = float(X_vec[0, idx])
                word_contributions.append({
                    "token": token,
                    "tfidf_value": round(tf_val, 4),
                    "weight": None,
                    "contribution_score": round(tf_val, 4),
                })

        # Sort descending by contribution score
        word_contributions.sort(key=lambda x: x["contribution_score"], reverse=True)
        top_positive = [w for w in word_contributions if w["contribution_score"] > 0][:5]
        top_opposing = [w for w in word_contributions if w["contribution_score"] < 0][-3:]

        # Standardized Ambiguity Analysis:
        # Flag ambiguous if margin between top-1 and top-2 is < 0.15 AND competing category probability is >= 0.15
        AMBIGUITY_MARGIN_THRESHOLD = 0.15
        AMBIGUITY_MIN_SECONDARY_PROB = 0.15

        sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        is_ambiguous = False
        rival_category = None
        confidence_margin = 1.0
        ambiguity_explanation = "Prediction shows high confidence with no significant class competition."

        if len(sorted_probs) >= 2:
            top1_cat, top1_prob = sorted_probs[0]
            top2_cat, top2_prob = sorted_probs[1]
            confidence_margin = round(top1_prob - top2_prob, 4)

            if confidence_margin < AMBIGUITY_MARGIN_THRESHOLD and top2_prob >= AMBIGUITY_MIN_SECONDARY_PROB:
                is_ambiguous = True
                rival_category = top2_cat
                ambiguity_explanation = (
                    f"Headline displays multi-domain characteristics: {confidence_margin:.1%} confidence margin "
                    f"between predicted '{top1_cat}' ({top1_prob:.1%}) and competing '{top2_cat}' ({top2_prob:.1%}). "
                    f"Vocabulary contains features with cross-domain overlap."
                )

        return {
            "top_contributing_tokens": top_positive,
            "opposing_tokens": top_opposing,
            "ambiguity_analysis": {
                "is_ambiguous": is_ambiguous,
                "confidence_margin": confidence_margin,
                "competing_category": rival_category,
                "explanation": ambiguity_explanation,
            },
            "interpretation_disclaimer": "Feature contributions reflect statistical corpus correlations (TF-IDF × class weights), not causal inferences."
        }
