
import torch

from enum import Enum

class Config:
    def __init__(self, args) -> None:

        # data hyperparameter
        self.batch_size = args.batch_size
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.max_seq_length = args.max_seq_length if "max_seq_length" in args.__dict__ else 0


        self.max_num_choices = args.max_num_choices if "max_num_choices" in args.__dict__ else 0
        self.max_candidate_length = args.max_candidate_length if "max_candidate_length" in args.__dict__ else 0

        self.num_workers = 8 ## workers

        # optimizer hyperparameter
        self.learning_rate = args.learning_rate
        self.max_grad_norm = args.max_grad_norm

        # training
        self.device = torch.device(args.device) # torch.cuda.set_device(2)


        self.num_epochs = args.num_epochs
        self.parallel = args.parallel if "parallel" in args.__dict__ else 0
        self.fp16 = args.fp16

        # model
        self.model_folder = args.model_folder
        self.bert_model_name = args.bert_model_name
        self.bert_folder = args.bert_folder
        self.height = args.height
        self.consider_multiple_m0 = args.consider_multiple_m0
        self.var_update_mode = args.var_update_mode

        self.use_constant = args.use_constant
        self.add_replacement = args.add_replacement

        self.train_file = args.train_file
        self.dev_file = args.dev_file
        self.test_file = args.test_file

        self.parallel = args.parallel if "parallel" in args.__dict__ else 0


        self.uni_labels = []