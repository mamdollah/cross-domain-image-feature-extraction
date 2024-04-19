# Cross-Domain Generalizability in Image Feature Extraction

This repository is maintained by two master's students, Mamdollah Amini and Adi Creson, who are currently researching the cross-domain generalizability of image feature extraction using large convolutional neural networks like ResNet50. The project aims to explore how networks trained for one task (such as image classification) can be repurposed as feature extractors in different domains, such as reinforcement learning.

## Research Overview

Our research investigates the potential of using ResNet50, a network originally designed for image classification, as a feature extractor in training reinforcement learning agents. This exploration involves assessing the generalizability and effectiveness of ResNet50 across various domains that it was not originally trained for. We are specifically evaluating the features extracted by the different convolutional blocks of ResNet50 to get a deeper understanding of their generalizability. This analysis helps determine which features are most useful and robust when applied outside the initial training context.

## Technologies Used

- **Python**: The primary programming language used for the project.
- **PyTorch**: A deep learning framework that provides a wide range of algorithms for deep learning.
- **Gymnasium**: An open-source library for developing and comparing reinforcement learning algorithms.
- **Stable Baselines3**: A set of reliable implementations of reinforcement learning algorithms in PyTorch.
- **Weights & Biases (wandb)**: A tool for tracking experiments, optimizing machine learning models, and reporting results.

## Requirements

Ensure you have all the necessary packages by installing them from the provided `requirements.txt` file:
pip install -r requirements.txt

## Ongoing Research

As this research is ongoing, we aim to regularly update this repository with our latest findings modifications to our methodology.

## Contact Information

For further inquiries or contributions, please feel free to contact:

- Mamdollah Amini - [mamdollah@gmail.com](mailto:mamdollah@gmail.com)
- Adi Creson - [adi@creson.com](mailto:adi@creson.com)
