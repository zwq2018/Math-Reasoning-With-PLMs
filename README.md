# Multi-View Reasoning: Consistent Contrastive Learning for Math Reasoning
This repo contains the code and data for our three papers about math reasoning:

- **EMNLP 2022 findings**: Multi-View Reasoning: Consistent Contrastive Learning for Math Reasoning    
[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2210.11694)
  
- **EMNLP2023 main**: An Expression Tree Decoding Strategy for Mathematical Equation Generation    
[![arxiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2310.09619)
  
- **IEEE/ACM Transactions on Audio, Speech, and Language Processing (TASLP)**: Specialized Mathematical Solving by a Step-By-Step Expression Chain Generation    
[![Taslp](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/abstract/document/10552332)

<img width="800" alt="image" src="https://user-images.githubusercontent.com/44236100/196947138-b54a139b-69bd-43c7-8cea-da1e61ddc829.png">



##🌿🌿 Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1

##🌿🌿 Description

- We provide the code for multilingual datasets ( math23k and mathQA )
- We also provide preprocessing code to process the equation by pre-order and post-order traversal
- We adopt Roberta-base and Chinese-BERT from HuggingFace for multilingual datasets. So it needs to be connected to the internet or copy the HuggingFace parameters from somewhere else and load them directly
- Part of bottom-up view code is based on Deductive-MWP (https://github.com/allanj/Deductive-MWP.git)

##🌿🌿 Usage

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
  
- The training code for the two datasets, MathQA and Math23k, differs slightly. Firstly, they utilize different pretrained models, with MathQA employing "roberta-base" instead of "roberta-chinese" for the English dataset. Secondly, MathQA has a different set of constants, including more constants such as '100.0', '1.0', '2.0', '3.0', '4.0', '10.0', '1000.0', '60.0', '0.5', '3600.0', '12.0', '0.2778', '3.1416', '3.6', '0.25', '5.0', '6.0', '360.0', '52.0', and '180.0'. Lastly, MathQA has two more operators compared to Math23k, with the additional operators being '^' and '^_rev'.


```
constants = ['100.0', '1.0', '2.0', '3.0', '4.0', '10.0', '1000.0', '60.0', '0.5', '3600.0', '12.0', '0.2778','3.1416', '3.6', '0.25', '5.0', '6.0', '360.0', '52.0', '180.0']
uni_labels = ['+', '-', '-_rev', '*', '/', '/_rev','^', '^_rev']
```

##🌿🌿 Details

From top-down view:
> <img width="800" alt="Xnip2023-08-26_18-27-18" src="https://github.com/zwq2018/Multi-view-Consistency-for-MWP/assets/44236100/bfab04d6-be3c-475e-ad21-261909e35abc">

From bottwom-view:
> <img width="800" alt="Xnip2023-08-26_18-27-38" src="https://github.com/zwq2018/Multi-view-Consistency-for-MWP/assets/44236100/6d17d2ff-8c7b-4e9c-83f6-2e1fef065fe2">


##🌿🌿 Citation
If you find this work useful, please cite our paper
```
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
    doi = "10.18653/v1/2022.findings-emnlp.79"
}

```
@inproceedings{zhang-etal-2023-expression,
    title = "An Expression Tree Decoding Strategy for Mathematical Equation Generation",
    author = "Zhang, Wenqi  and
      Shen, Yongliang  and
      Nong, Qingpeng  and
      Tan, Zeqi  and
      Ma, Yanna  and
      Lu, Weiming",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.29",
    doi = "10.18653/v1/2023.emnlp-main.29",
}
```
