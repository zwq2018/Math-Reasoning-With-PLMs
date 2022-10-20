import traceback

import torch
from torch.utils.data import Dataset
from typing import List, Union
from transformers import PreTrainedTokenizerFast, BertTokenizerFast, BertTokenizer, RobertaTokenizer, RobertaTokenizerFast
from tqdm import tqdm
from torch.utils.data._utils.collate import default_collate
import numpy as np
from src.utils import read_data, write_data
import collections
import re
from src.eval.utils import compute_value, compute_value_for_incremental_equations, compute_value_for_parallel_equations
import math
from typing import Dict, List
from collections import Counter
import logging
from copy import deepcopy
from src.data.lang import Lang
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class_name_2_quant_list = {
    "bert-base-cased": ['<', 'q', '##uant', '>'],
    "roberta-base": ['Ġ<', 'quant', '>'],
    "bert-base-multilingual-cased": ['<', 'quant', '>'],
    "xlm-roberta-base": ['▁<', 'quant', '>'],
    'bert-base-chinese': ['<', 'q', '##uan', '##t', '>'],
    'hfl/chinese-bert-wwm-ext': ['<', 'q', '##uan', '##t', '>'],
    'hfl/chinese-roberta-wwm-ext': ['<', 'q', '##uan', '##t', '>'],
}

UniFeature = collections.namedtuple('UniFeature', 'input_ids attention_mask token_type_ids variable_indexs_start variable_indexs_end num_variables variable_index_mask labels label_height_mask target_len target_idx RE_tree_align seq_infix_idx seq_infix_len num_list')
UniFeature.__new__.__defaults__ = (None,) * 14

class UniversalDataset(Dataset):

    def __init__(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 uni_labels:List[str],
                 pretrained_model_name:str,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None,
                 constant2id: Dict[str, int] = None,
                 constant_values: List[float] = None,
                 use_incremental_labeling: bool = False,
                 data_max_height: int = 100,
                 test_strings: List[str] = None) -> None:
        self.tokenizer = tokenizer
        self.constant2id = constant2id
        self.constant_values = constant_values
        self.constant_num = len(self.constant2id) if self.constant2id else 0
        self.use_incremental_labeling = use_incremental_labeling
        self.add_replacement = add_replacement
        self.data_max_height = data_max_height
        self.uni_labels = uni_labels
        self.quant_list = class_name_2_quant_list[pretrained_model_name]
        self.output_lang  = Lang()
        self.output_lang.build_output_lang_for_tree()
        self.max_infix_len  = -1
        filtered_steps = [int(v) for v in filtered_steps] if filtered_steps is not None else None
        if file is not None:
            self.read_math23k_file(file, tokenizer, number, add_replacement, filtered_steps)
        else:
            self._features = []
            self.insts = []
            for sent, num_list in test_strings:
                for k in range(ord('a'), ord('a') + 26):
                    sent = sent.replace(f"temp_{chr(k)}", " <quant> ")
                res = tokenizer.encode_plus(" " + sent, add_special_tokens=True, return_attention_mask=True)

                input_ids = res["input_ids"]
                attention_mask = res["attention_mask"]
                tokens = tokenizer.convert_ids_to_tokens(input_ids)
                var_starts = []
                var_ends = []
                quant_num = len(self.quant_list)
                for k, token in enumerate(tokens):
                    if (token == self.quant_list[0]) and tokens[k:k + quant_num] == self.quant_list:
                        var_starts.append(k)
                        var_ends.append(k + quant_num - 1)
                num_variable = len(var_starts)
                var_mask = [1] * len(var_starts)
                labels = [[-100, -100, -100, -100]]
                label_height_mask = [0]
                self.insts.append({"sent": sent, "num_list":num_list})
                self._features.append(
                    UniFeature(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=[0] * len(input_ids),
                               variable_indexs_start=var_starts,
                               variable_indexs_end=var_ends,
                               num_variables=num_variable,
                               variable_index_mask=var_mask,
                               labels=labels,
                               label_height_mask=label_height_mask)
                )


    def read_math23k_file(self, file: Union[str, None],
                 tokenizer: PreTrainedTokenizerFast,
                 number: int = -1, add_replacement: bool = False,
                 filtered_steps: List = None) -> None:
        data = read_data(file=file)
        if number > 0:
            data = data[:number]
        # ## tokenization
        self._features = []
        max_num_steps = 0
        self.insts = []
        num_step_count = Counter()
        equation_layer_num = 0
        equation_layer_num_count = Counter()
        var_num_all =0
        var_num_count = Counter()
        sent_len_all = 0
        filter_type_count = Counter()
        found_duplication_inst_num = 0
        filter_step_count = 0
        for obj in tqdm(data, desc='Tokenization', total=len(data)):
            if obj['type_str'] != "legal" and obj['type_str'] != "variable more than 7":      #
                if "^" in self.uni_labels and obj["type_str"] == "have square":
                    pass
                else:
                    filter_type_count[obj["type_str"]] += 1
                    continue
            mapped_text = obj["text"]
            sent_len = len(mapped_text.split())
            for k in range(ord('a'), ord('a') + 26):
                mapped_text = mapped_text.replace(f"temp_{chr(k)}", " <quant> ")
            if "math23k" in file:
                mapped_text = mapped_text.split()
                input_text = ""
                for idx, word in enumerate(mapped_text):
                    if word.strip() == "<quant>":
                        input_text += " <quant> "
                    elif word == "," or word == "，":
                        input_text += word + " "
                    else:
                        input_text += word
            else:
                raise NotImplementedError("The file type is not supported")
            res = tokenizer.encode_plus(" " + input_text, add_special_tokens=True, return_attention_mask=True)
            input_ids = res["input_ids"]
            attention_mask = res["attention_mask"]
            tokens = tokenizer.convert_ids_to_tokens(input_ids)
            var_starts = []
            var_ends = []
            quant_num = len(self.quant_list)

            for k, token in enumerate(tokens):
                if (token == self.quant_list[0]) and tokens[k:k + quant_num] == self.quant_list:
                    var_starts.append(k)
                    var_ends.append(k + quant_num - 1)

            assert len(input_ids) < 512
            num_variable = len(var_starts)
            assert len(var_starts) == len(obj["num_list"])
            if len(obj["num_list"]) == 0:
                filter_type_count["no detected variable"] += 1
                obj['type_str'] = "no detected variable"
                continue
            var_mask = [1] * num_variable
            if len(obj["equation_layer"])  == 0:
                filter_type_count["empty equation in the data"]  += 1
                obj['type_str'] = "empty eqution"
                continue


            if "nodup" in file:
                eq_set = set()
                for equation in obj["equation_layer"]:
                    eq_set.add(' '.join(equation))
                try:
                    assert len(eq_set) == len(obj["equation_layer"])
                except:
                    found_duplication_inst_num += 1

            if self.use_incremental_labeling:
                labels = self.get_label_ids_incremental(obj["equation_layer"], add_replacement=add_replacement)
                target_eq = obj["target_template"][2:]
                target_eq_prefix = self.from_infix_to_prefix(target_eq)
                if len(target_eq_prefix)==1 :
                    target_eq_prefix = ["*", "1"] + target_eq_prefix
                    target_eq = target_eq + ["*", "1"]

                target_len = len(target_eq_prefix)
                target_eq_prefix_idx = self.prepare_label_tree_decoder(self.output_lang, target_eq_prefix)
                align = obj["align"]
                seq_infix_len = len(target_eq)
                target_eq_infix = [e[-1] if e.startswith("temp_") else e for e in target_eq ]
                seq_infix_idx = self.prepare_label_tree_decoder(self.output_lang, target_eq_infix)
                num_list = obj['num_list']
                if seq_infix_len > self.max_infix_len:
                    self.max_infix_len = seq_infix_len

                assert  len(align) == target_len
                if 28 in target_eq_prefix_idx or 29 in seq_infix_idx:
                    print(obj['original_text'])
                    print(target_eq)
                    print(target_eq_prefix)



            else:
                labels = self.get_label_ids_updated(obj["equation_layer"], add_replacement=add_replacement)

            if not labels:
                filter_type_count["cannot obtain the label sequence"] += 1
                obj['type_str'] = "illegal"
                continue
            # compute_value(labels, obj["num_list"])

            if len(labels) > self.data_max_height:
                filter_type_count[f"larger than the max height {self.data_max_height}"] += 1
                continue
            for left, right, _, _ in labels:
                assert left <= right

            if isinstance(labels, str):
                filter_type_count[f"index error for labels"] += 1
                obj['type_str'] = "illegal"
                continue
            try:
                if self.use_incremental_labeling:
                    res, _ = compute_value_for_incremental_equations(labels, obj["num_list"], self.constant_num, uni_labels=self.uni_labels, constant_values=self.constant_values)
                else:
                    res = compute_value(labels, obj["num_list"], self.constant_num, uni_labels=self.uni_labels, constant_values=self.constant_values)
            except:
                # print("answer calculate exception")
                filter_type_count[f"answer_calculate_exception"] += 1
                obj['type_str'] = "illegal"
                continue

            diff = res - float(obj["answer"])
            try:
                if float(obj["answer"]) > 1000000:
                    assert math.fabs(diff) < 200
                else:
                    assert math.fabs(diff) < 1
            except:
                # traceback.print_exc()
                obj['type_str'] = "illegal"
                if "test" in file or "valid" in file:
                    filter_type_count[f"answer not equal"] += 1
                    continue
            if filtered_steps is not None:
                if len(labels) not in filtered_steps:
                    filter_step_count += 1
                    continue

            label_height_mask = [1] * len(labels)
            num_step_count[len(labels)] += 1                      #
            max_num_steps = max(max_num_steps, len(labels))
            ## check label all valid
            for label in labels:
                assert all([label[i] >= 0 for i in range(4)])
            equation_layer_num += len(obj["equation_layer"])
            equation_layer_num_count[len(obj["equation_layer"])] += 1
            sent_len_all += sent_len
            var_num_all += len(obj["num_list"])
            var_num_count[len(obj["num_list"])] += 1
            self._features.append(
                UniFeature(input_ids=input_ids,
                           attention_mask=attention_mask,
                           token_type_ids = [0] * len(input_ids),
                           variable_indexs_start=var_starts,
                           variable_indexs_end=var_ends,
                           num_variables=num_variable,
                           variable_index_mask=var_mask,
                           labels = labels,
                           label_height_mask=label_height_mask,
                           target_len = target_len,
                           target_idx = target_eq_prefix_idx,
                           RE_tree_align = align,
                           seq_infix_idx = seq_infix_idx,
                           seq_infix_len = seq_infix_len,
                           num_list = num_list
                           )
            )
            self.insts.append(obj)
        logger.info(f", total number instances: {len(self._features)} (before filter: {len(data)}), max num steps: {max_num_steps}")
        self.number_instances_remove = sum(filter_type_count.values())
        logger.info(f"filtered type counter: {filter_type_count}")
        logger.info(f"number of instances removed: {self.number_instances_remove}")
        assert self.number_instances_remove == len(data) - len(self._features)
        if found_duplication_inst_num:
            logger.warning(f"[WARNING] find duplication num: {found_duplication_inst_num} (not removed)")
        logger.debug(f"filter step count: {filtered_steps}")
        logger.info(num_step_count)
        avg_eq_num = equation_layer_num * 1.0/ len(self._features)
        logger.debug(f"average operation number: {avg_eq_num}, total: {equation_layer_num}, counter: {equation_layer_num_count}")
        avg_sent_len = sent_len_all * 1.0 / len(self._features)
        logger.debug(f"average sentence length: {avg_sent_len}, total: {sent_len_all}")
        logger.debug(f"variable number avg: {var_num_all * 1.0 / len(self._features)}, total: {var_num_all}, counter:{var_num_count}")

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx) -> UniFeature:
        return self._features[idx]

    def indexes_from_sentence(self, lang, sentence, tree=False):
        res = []
        for word in sentence:
            if word =='PI':
                word ='3.14'
            if word =='1':
                word = '1.0'
            if len(word) == 0:
                continue
            if word in lang.word2index:
                res.append(lang.word2index[word])
            else:
                print('22222222222222 unk word',word )
                res.append(lang.word2index["UNK"])
        if "EOS" in lang.index2word and not tree:
            res.append(lang.word2index["EOS"])
        return res


    def from_infix_to_prefix(self,exp):
        st = list()
        res = list()
        priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
        expression = deepcopy(exp)
        expression.reverse()
        for e in expression:
            if e in [")", "]"]:
                st.append(e)
            elif e == "(":
                c = st.pop()
                while c != ")":
                    res.append(c)
                    c = st.pop()
            elif e == "[":
                c = st.pop()
                while c != "]":
                    res.append(c)
                    c = st.pop()
            elif e in priority:
                while len(st) > 0 and st[-1] not in [")", "]"] and priority[e] < priority[st[-1]]:
                    res.append(st.pop())
                st.append(e)
            elif e.startswith("temp_"):
                res.append(e[-1])
            else:
                res.append(e)
        while len(st) > 0:
            res.append(st.pop())
        res.reverse()
        return res
    def prepare_label_tree_decoder(self, output_lang, equ_pre):
        output_lang.add_sen_to_vocab(equ_pre)
        output_idx = self.indexes_from_sentence(output_lang, equ_pre, True)
        return output_idx



    def get_label_ids(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        label_ids = []
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0

            left_var_idx = (ord(left_var) - ord('a')) if left_var != "#" else -1
            right_var_idx = (ord(right_var) - ord('a'))
            assert right_var_idx >= 0
            try:
                assert left_var_idx >=0 or left_var_idx == -1
            except:
                return "index error"
            if left_var_idx < right_var_idx:
                op_idx = self.uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                # left > right
                if op in ["+", "*"]:
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    assert not op.endswith("_rev")
                    op_idx = self.uni_labels.index(op + "_rev")
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


    def get_label_ids_updated(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        # in this data, only have one or zero bracket
        label_ids = []
        num_constant = len(self.constant2id) if self.constant2id is not None else 0
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0

            if left_var != "#" and (not left_var.startswith("m_")):
                if self.constant2id is not None and left_var in self.constant2id:
                    left_var_idx = self.constant2id[left_var]
                else:
                    # try:
                    assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    # except:
                    #     print("seohting")
                    left_var_idx = (ord(left_var) - ord('a') + num_constant)
            else:
                left_var_idx = -1
            right_var_idx = (ord(right_var) - ord('a') + num_constant) if self.constant2id is None or (right_var not in self.constant2id) else self.constant2id[right_var]
            # try:
            assert right_var_idx >= 0
            # except:
            #     print("right var index error")
            #     return "right var index error"
            # try:
            assert left_var_idx >= -1
            # except:
            #     return "index error"
            if left_var_idx <= right_var_idx:
                if left_var_idx == right_var_idx and op.endswith("_rev"):
                    op = op[:-4]
                op_idx = self.uni_labels.index(op)
                label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                # left > right
                if (op in ["+", "*"]):
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    # assert not op.endswith("_rev")
                    op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else  self.uni_labels.index(op[:-4])
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


    def get_label_ids_incremental(self, equation_layers: List, add_replacement: bool) -> Union[List[List[int]], None]:
        label_ids = []
        num_constant = len(self.constant2id) if self.constant2id is not None else 0
        for l_idx, layer in enumerate(equation_layers):
            left_var, right_var, op = layer
            if left_var == right_var and (not add_replacement):
                return None
            is_stop = 1 if l_idx == len(equation_layers) - 1 else 0
            if (not left_var.startswith("m_")):
                if self.constant2id is not None and left_var in self.constant2id:
                    left_var_idx = self.constant2id[left_var] + l_idx
                else:
                    try:
                        assert ord(left_var) >= ord('a') and ord(left_var) <= ord('z')
                    except:
                        print(f"[WARNING] find left_var ({left_var}) invalid, returning FALSE")
                        return None
                    left_var_idx = (ord(left_var) - ord('a') + num_constant + l_idx)
            else:
                m_idx = int(left_var[2:])
                # left_var_idx = -1
                left_var_idx = l_idx - m_idx
            if (not right_var.startswith("m_")):
                if self.constant2id is not None and right_var in self.constant2id:
                    right_var_idx = self.constant2id[right_var] + l_idx
                else:
                    try:
                        assert ord(right_var) >= ord('a') and ord(right_var) <= ord('z')
                    except:
                        print(f"[WARNING] find right var ({right_var}) invalid, returning FALSE")
                        return None
                    right_var_idx = (ord(right_var) - ord('a') + num_constant + l_idx)
            else:
                m_idx = int(right_var[2:])
                # left_var_idx = -1
                right_var_idx = l_idx - m_idx
            # try:
            assert right_var_idx >= 0
            # except:
            #     print("right var index error")
            #     return "right var index error"
            # try:
            assert left_var_idx >= 0
            # except:
            #     return "index error"

            if left_var.startswith("m_") or right_var.startswith("m_"):
                if left_var.startswith("m_") and (not right_var.startswith("m_")):
                    assert left_var_idx < right_var_idx
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                elif not left_var.startswith("m_") and right_var.startswith("m_"):
                    assert left_var_idx > right_var_idx
                    op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4]) #
                    label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                else:
                    if left_var_idx >= right_var_idx:
                        op = op[:-4] if left_var_idx == right_var_idx and op.endswith("_rev") else op
                        op_idx = self.uni_labels.index(op)
                        if left_var_idx > right_var_idx and (op not in ["+", "*"]):
                            op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:
                        if (op in ["+", "*"]):
                            op_idx = self.uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                        else:
                            assert  "+" not in op and "*" not in op
                            op_idx = self.uni_labels.index(op)
                            label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
            else:
                if left_var_idx <= right_var_idx:
                    if left_var_idx == right_var_idx and op.endswith("_rev"):
                        op = op[:-4]
                    op_idx = self.uni_labels.index(op)
                    label_ids.append([left_var_idx, right_var_idx, op_idx, is_stop])
                else:
                    if (op in ["+", "*"]):
                        op_idx = self.uni_labels.index(op)
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
                    else:

                        op_idx = self.uni_labels.index(op + "_rev") if not op.endswith("_rev") else self.uni_labels.index(op[:-4])
                        label_ids.append([right_var_idx, left_var_idx, op_idx, is_stop])
        return label_ids


    def collate_function(self, batch: List[UniFeature]):
        max_wordpiece_length = max([len(feature.input_ids)  for feature in batch])
        max_num_variable = max([feature.num_variables  for feature in batch])
        max_height = max([len(feature.labels) for feature in batch])
        max_target_len = max([feature.target_len for feature in batch])
        max_seq_len = max([feature.seq_infix_len for feature in batch])
        padding_value = [-1, 0, 0, 0] if not self.use_incremental_labeling else [0,0,0,0]
        if self.use_incremental_labeling and not self.add_replacement:
            padding_value = [0, 1, 0, 0]
        for i, feature in enumerate(batch):
            padding_length = max_wordpiece_length - len(feature.input_ids)
            input_ids = feature.input_ids + [self.tokenizer.pad_token_id] * padding_length
            attn_mask = feature.attention_mask + [0]* padding_length
            token_type_ids = feature.token_type_ids + [0]* padding_length
            padded_variable_idx_len = max_num_variable - feature.num_variables
            var_starts = feature.variable_indexs_start + [0] * padded_variable_idx_len
            var_ends = feature.variable_indexs_end + [0] * padded_variable_idx_len
            variable_index_mask = feature.variable_index_mask + [0] * padded_variable_idx_len
            padded_height = max_height - len(feature.labels)
            labels = feature.labels + [padding_value]* padded_height
            label_height_mask = feature.label_height_mask + [0] * padded_height
            target_len = feature.target_len
            target_idx = feature.target_idx + [0] * (max_target_len-feature.target_len)
            RE_tree_align = feature.RE_tree_align + [-2] * (max_target_len-feature.target_len)

            seq_infix_len = feature.seq_infix_len
            seq_infix_idx = feature.seq_infix_idx + [0] * (max_seq_len-feature.seq_infix_len)
            assert  len(feature.num_list) == len(feature.variable_indexs_start)
            num_list_batch =  feature.num_list + [ 1.234 ] * padded_variable_idx_len

            batch[i] = UniFeature(input_ids=np.asarray(input_ids),
                                attention_mask=np.asarray(attn_mask),
                                  token_type_ids=np.asarray(token_type_ids),
                                 variable_indexs_start=np.asarray(var_starts),
                                 variable_indexs_end=np.asarray(var_ends),
                                 num_variables=np.asarray(feature.num_variables),
                                 variable_index_mask=np.asarray(variable_index_mask),
                                 labels =np.asarray(labels),
                                  label_height_mask=np.asarray(label_height_mask),
                                  target_len = np.asarray(target_len),
                                  target_idx = np.asarray(target_idx),
                                  RE_tree_align= np.asarray(RE_tree_align),
                                  seq_infix_idx = np.asarray(seq_infix_idx),
                                  seq_infix_len = np.asarray(seq_infix_len),
                                  num_list = np.asarray(num_list_batch))
        results = UniFeature(*(default_collate(samples) for samples in zip(*batch)))
        return results


def main_for_math23k():
    pretrained_language_model = 'hfl/chinese-roberta-wwm-ext'
    tokenizer = BertTokenizer.from_pretrained(pretrained_language_model)
    constant2id = {"1": 0, "PI": 1}
    constant_values = [1.0, 3.14]
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    data_max_height = 15
    UniversalDataset(file="../../data/math23k/test23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, pretrained_model_name=pretrained_language_model,
                     data_max_height = data_max_height)
    UniversalDataset(file="../../data/math23k/train23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height, filtered_steps=None)
    UniversalDataset(file="../../data/math23k/valid23k_processed_nodup.json", tokenizer=tokenizer, uni_labels=uni_labels,
                     constant2id=constant2id, constant_values=constant_values, add_replacement=add_replacement,
                     use_incremental_labeling=use_incremental_labeling, pretrained_model_name=pretrained_language_model,
                     data_max_height=data_max_height)

if __name__ == '__main__':
    logger.addHandler(logging.StreamHandler())
    from transformers import BertTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast
    add_replacement = True
    use_incremental_labeling = True
    class_name_2_tokenizer = {
        "bert-base-cased": BertTokenizerFast,
        "roberta-base": RobertaTokenizerFast,
        "bert-base-multilingual-cased": BertTokenizerFast,
        "xlm-roberta-base": XLMRobertaTokenizerFast,
        'hfl/chinese-bert-wwm-ext': BertTokenizerFast,
        'hfl/chinese-roberta-wwm-ext': BertTokenizerFast,
    }

    main_for_math23k()



