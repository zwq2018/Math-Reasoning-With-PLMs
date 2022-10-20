## Multi-view Math Solving 

### Requirements
* transformers `pip3 install transformers`
* Pytorch > 1.7.1

### Description

- We provide the code for multilingual datasets ( math23k and mathQA )
- We also provide preprocessing code to process the equation by pre-order and post-order traversal
- We adopt Roberta-base and Chinese-BERT from HuggingFace for multilingual datasets. So it needs to be connected to the internet or copy the HuggingFace parameters from somewhere else and load them directly

### Usage

- To reproduce our results, you can either directly use the dataset we provided, where we have processed the equation using pre-order and post-order traversal, or directly process the original dataset using our code

- To run the code for math23k in the corresponding folder:   

  ```
  python main_math23k.py
  ```

- To preprocess the original dataset:

  ```
  cd preprocess
  python process_math23k.py
  ```

  

