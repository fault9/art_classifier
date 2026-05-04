# Art Movement Classification with CLIP Embeddings

## Links

GitHub repository: https://github.com/fault9/art_classifier

Hugging Face dataset: https://huggingface.co/datasets/fault9/art-movement-rebalanced-dataset

Hugging Face model: https://huggingface.co/fault9/art-movement-clip-classifier

Hugging Face demo: https://huggingface.co/spaces/fault9/art-movement-classifier-demo

## Introduction

This project investigates whether pretrained image embeddings can support art movement classification on a small custom painting dataset. The domain problem is that art movement labels are visually and historically overlapping: Renaissance and Baroque share religious subjects, Cubism and Abstract art share geometric forms, and Expressionism overlaps with Abstract Expressionism. A second problem is dataset bias. If one artist dominates a class, a classifier may learn artist-specific style rather than movement-level visual structure.

The project therefore uses CLIP image embeddings as a compact visual representation and trains shallow classifiers on top of these frozen embeddings. The final dataset contains 1,478 paintings across eight classes: Renaissance, Baroque, Impressionism, Expressionism, Cubism, Abstract, Surrealism, and Pop Art. The dataset was assembled from WikiArt-derived metadata and Hugging Face/WikiArt sources, then rebalanced by capping dominant artists and adding underrepresented artists where possible.

## Method

Images were resized and represented using `openai/clip-vit-base-patch32`, producing 512-dimensional image embeddings. Several classifiers were evaluated on the embeddings, including linear and non-linear scikit-learn models. The final selected model was an MLP classifier trained on CLIP embeddings. Evaluation used a fixed train/test split with 1,090 training images and 388 held-out test images. The training script also performs cross-validation to compare model choices.

The web demo is implemented in Gradio and deployed on Hugging Face Spaces. It supports single-image classification, confidence visualization, batch collection analysis, and two interpretability layers. The Wölfflin view maps the predicted movement to theoretical art-history axes, mainly as a Renaissance-Baroque reference because that is the historical contrast Wölfflin developed. The Arnheim view projects the image embedding onto perceptual axes constructed from anchor paintings, such as balance, depth, light, and color. Its nearest profile is interpreted as perceptual resemblance rather than a second class prediction.

## Results

The final model achieved a held-out test accuracy of **0.8093** and a cross-validation accuracy of **0.8138 ± 0.0173**. The strongest classes were Impressionism and Renaissance, both above 0.90 F1. Baroque also performed well at 0.86 F1. Surrealism and Abstract were more difficult, reflecting the visual diversity of these categories and their overlap with other modern movements. The interpretability layers also showed that movement labels and perceptual structure are related but not identical: a painting may be classified as Cubism while its Arnheim scores sit near Surrealism or Pop Art. This is treated as useful evidence of visual overlap, not as an error in the classifier.

The project satisfies the assignment requirements by including a custom dataset, a trained classifier specific to art movement classification, and a working Hugging Face web demo. The dataset repository is intended to be private or gated because the source images may have mixed copyright status. The model and Space can remain public because they do not redistribute the training images or the cached per-image embedding matrix.

## Reflection on AI Assistance

AI coding tools were useful for speeding up implementation, writing scripts, generating audit tables, debugging deployment issues, and preparing Hugging Face model and dataset cards. They were especially helpful in identifying edge cases around artist dominance, stale embedding caches, and dataset rebalancing. 

However, the work required manual checking. The AI sometimes made assumptions that needed correction or additional filtering, such as initially treating WikiArt availability as if it implied copyright freedom, or not balancing the dataset sufficiently across artists. Model outputs also required domain judgment: for example, a Caravaggio painting may be classified as Renaissance if its composition is calmer and more classical than the Baroque examples in the training set. Similarly, Arnheim nearest-profile results should not be read as movement labels; they describe perceptual similarity on a small set of axes. This reinforced that AI assistance is productive for implementation, but the final dataset design, interpretation, copyright framing, and evaluation claims must be checked by the developer.

## References

Wölfflin, H. (1915). *Principles of Art History*.

Arnheim, R. (1954). *Art and Visual Perception: A Psychology of the Creative Eye*.
