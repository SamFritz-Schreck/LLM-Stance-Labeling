# Stance Classification with Large Language Models

This repository contains code and analysis for the paper **"Prompting and Fine-Tuning Open-Sourced Large Language Models for Stance Classification"**, under review in ACM TIST. The paper investigates the use of open-source Large Language Models (LLMs) for stance classification tasks, comparing various prompting schemes and fine-tuning methods.

## Files in this Repository

- **`fine_tune_model.py`**: Code for fine-tuning open-source LLMs on stance classification datasets.
- **`label_stance.py`**: Script for performing stance classification using LLMs using various pormpting schemes, including zero-shot or few-shot prompting.
- **`label_stance_by_fine_tuned_model.py`**: Script for applying fine-tuned LLMs to classify stances.
- **`Stance Labeling Analysis.ipynb`**: Jupyter notebook for analyzing results and visualizing stance classification metrics.

## Methodology

This repository supports the following key workflows:
1. **Prompting Schemes**: Employ various prompting schemes (e.g., task-based, zero-shot, Chain-of-Thought) for stance classification without additional fine-tuning.
2. **Fine-Tuning**: Fine-tune LLMs for specific stance classification tasks using benchmark datasets.
3. **Evaluation**: Compare performance across models and prompting schemes using unweighted macro F1-scores.
