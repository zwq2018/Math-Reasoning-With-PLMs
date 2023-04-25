# Multi-View Reasoning: Consistent Contrastive Learning for Math Word Problem.
This repo contains the code and data for our EMNLP 2022 findings paper  

<img width="456" alt="image" src="https://user-images.githubusercontent.com/44236100/196947138-b54a139b-69bd-43c7-8cea-da1e61ddc829.png">



## Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1

## Description

- We provide the code for multilingual datasets ( math23k and mathQA )
- We also provide preprocessing code to process the equation by pre-order and post-order traversal
- We adopt Roberta-base and Chinese-BERT from HuggingFace for multilingual datasets. So it needs to be connected to the internet or copy the HuggingFace parameters from somewhere else and load them directly
- Part of bottom-up view code is based on Deductive-MWP (https://github.com/allanj/Deductive-MWP.git)

## Usage

- To reproduce our results, you can either directly use the dataset we provided, where we have processed the equation using pre-order and post-order traversal, or directly process the original dataset using our code

- To run the code for math23k in the corresponding folder:   

  ```
  python main_math23k.py
  ```

- To preprocess the original dataset for equation augmentation:

  ```
  cd preprocess
  python process_math23k.py
  ```
## Citation
If you find this work useful, please cite our paper



@inproceedings{zhang-etal-2022-multi-view,
    title = "Multi-View Reasoning: Consistent Contrastive Learning for Math Word Problem",
    author = "Zhang, Wenqi  and
      Shen, Yongliang  and
      Ma, Yanna  and
      Cheng, Xiaoxia  and
      Tan, Zeqi  and
      Nong, Qingpeng  and
      Lu, Weiming",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2022",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.findings-emnlp.79",
    pages = "1103--1116"
}


  

