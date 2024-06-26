---
permalink: /
title: "Little About Me, More About My GenAI Projects"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

## About Me
I earned my PhD in Materials Science and a Master's degree in Computer Science from Duke University in 2022. During my doctoral studies, I specialized in conducting simulations on supercomputers to explore various material properties. My Master's program focused on machine learning, deep learning, and natural language processing techniques. This dual expertise equips me with a strong foundation in both theoretical research and practical applications across interdisciplinary domains. 

In addition to my academic background, I've gained practical experience in VQGAN, diffuse models and natural language processing projects (details about projects will be posted in [portfolio](https://syang-lab.github.io/homepage/portfolio/)--comming soon). 

Currently, I'm in the process of writing a comprehensive blog series on diffuse models. This is the first of the series [blog post 1](https://syang-lab.github.io/homepage/posts/2024-02-14-Diffuse-Model-Blog-Series1), diving into the underlying physics and mathematics behind diffuse model. The second [blog post 2](https://syang-lab.github.io/homepage/posts/2024-03-3-Diffuse-Model-Blog-Series2) is on the way, convering zero-shot image editing techniques based on diffuse model.

## Projects
* Text-to-SQL Chat Demo based on Instruction Fine-tuned Llama (7B) Model [GitHub](https://syang-lab.github.io/Text2SQL-RAG-Chatbot/)
  * Enhanced efficiency of Llama model instruction fine-tuning by integrating quantization, LoRA, and deepSpeed methods.
  * Reduce the trainable parameters by 60%, enabling the instruction fine-tuning of the Llama model on a 30% A100 with 24G memory.
  * Constructed a vector database to enhance Llama’s performance utilizing the Retrieval-Augmented Generation (RAG) technique and the Langchain framework. Additionally, developed a Gradio chat web demo.
* Implemented Layered Rendering Diffusion Model for Zero-Shot Guided Image Synthesis [Colab LRDiff](https://colab.research.google.com/drive/1KcNvrjh7k5G4FFbzeMfdGruA-o0Y4XZB)
  * Launched the first version of the rendering diffusion model, enabling the synthesis of images that adhere to particular spatial and contextual specifications. Published the code to paperwithcode.
  * LRDiff can manipulate the spatial arrangement of semantic objects via sampling without the necessity for retraining.
  * Vision guidance was incorporated through introducing perturbations to the sampling process using weighted masks. The weights of masks were determined by the intensity of cross-attention maps between input masks and semantic objects in the Unet.

* Implemented Pathway Autoregressive Text to Image (Parti) Model [github Parti](https://github.com/syang-lab/Pathway_Autoregressive_Text2Image_Model)
  * Published an unofficial implementation of a two stage Parti model. Stage1 contains ViT-VQGAN, while Stage2
integrates ViT-VQGAN with transformer encoder and decoder.
  * Presented the initial unofficial release of the Parti model, emphasizing the integration of all modules and
training codes within a unified package, addressing the absence of training code in existing Parti implementations.

* Financial Event Detection and Text Summarization [github NLP](https://github.com/syang-lab/NLP_Project)
  * Constructed and trained NLP models to evaluate the significance of domain adaptation in financial event classification and financial text summarization tasks, utilizing a financial dataset.
  * Compared DistilBert-based (66M Params) event detection models with and without masked language modeling adaptation. The inclusion of domain adaptation led to a 6.13% increase in single-shot accuracy and a 5.28% improvement in fine-tuning accuracy.
  * Implemented and assessed the performance of t5-small (580M Params) on summarization tasks, employing both with and without denoising adaptation. Utilized ROUGE scores for evaluation, observing about 1.4 improvements in few-shot rougeL and rougeLsum scores with the inclusion of denoising adaptation.
  * Used SageMaker for training, evaluation, hyperparameter tuning and deployment. Created Gradio apps for financial event detection and text summarization, and deployed them to Hugging Face Spaces. [Financialevent_Detection](https://huggingface.co/spaces/SHSH0819/event_detection_app) and [FinancialNews_Summarization](https://huggingface.co/spaces/SHSH0819/FinancialNews_Summarization_APP)

* Created Generation Evaluation Matrix Frechet CLIP Distance [github FCD_Score](https://github.com/syang-lab/FCD_Score)
  * To overcome the domain limitations of the FID score, I created a new evaluation matrix called Frechet CLIP Distance. This metric depends on the pretrained CLIP model and is distributed through PyPI.

## Research Experience
* Research Assistant, Duke University, 2016-2022
  * Published papers in Science and Nature Communications as co-first author.
  * Developed a python package capable of computing time-resolved experimental signals, featuring an implementation of the auto-correlation function to simulate spectral signals arising from atom vibrations.
  * Conducted an extensive examination of the advancements and obstacles in integrating computational materials science with deep learning, documented in my PhD thesis.

## Peer Review Experience
* International Conference on Learning Representations
* Computer Physics Communications
* Physical Review Materials
* MRS Advances
