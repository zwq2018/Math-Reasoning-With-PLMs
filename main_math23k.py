from src.data.universal_dataset import UniversalDataset
from src.config import Config
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, PreTrainedTokenizer, RobertaTokenizerFast, XLMRobertaTokenizerFast
from tqdm import tqdm
import argparse
from src.utils import get_optimizers, write_data
import torch
import torch.nn as nn
import numpy as np
import copy
import os
import random
from src.model.universal_model import UniversalModel
from collections import Counter
from src.eval.utils import is_value_correct, compute_prefix_tree_result
from typing import List, Tuple
import logging
from transformers import set_seed
from torch.utils.tensorboard import SummaryWriter
import warnings
import time


warnings.filterwarnings('ignore')
runtime = time.strftime("%Y-%m-%d %H:%M:%S")

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logging.basicConfig(
	format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
	datefmt="%m/%d/%Y %H:%M:%S",
	level=logging.INFO,
)
torch.cuda.empty_cache()
class_name_2_model = {
        'hfl/chinese-roberta-wwm-ext': UniversalModel,
    }
path = './path/to/log'
writer = SummaryWriter(os.path.join(path,str(runtime)))


def parse_arguments(parser:argparse.ArgumentParser):
    # data Hyperparameters
    parser.add_argument('--device', type=str, default="cuda:0", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'], help="GPU/CPU devices")
    parser.add_argument('--batch_size', type=int, default = 12 )
    parser.add_argument('--train_num', type=int, default = -1, help="The number of training data, -1 means all data")
    parser.add_argument('--dev_num', type=int, default = -1, help="The number of development data, -1 means all data")
    parser.add_argument('--test_num', type=int, default = -1, help="The number of development data, -1 means all data")


    parser.add_argument('--train_file', type=str, default="data/math23k/combined_train23k_processed_multi_view_align.json")
    parser.add_argument('--dev_file', type=str, default="data/math23k/test23k_processed_multi_view_align.json")
    parser.add_argument('--test_file', type=str, default="data/math23k/test23k_processed_multi_view_align.json")

    parser.add_argument('--train_filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--test_filtered_steps', default=None, nargs='+', help="some heights to filter")
    parser.add_argument('--use_constant', default=1, type=int, choices=[0, 1], help="whether to use constant 1 and pi")

    parser.add_argument('--add_replacement', default=1, type=int, choices=[0,1], help = "use replacement when computing combinations")

    # model
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--model_folder', type=str, default="math23k_chinese-roberta-wwm-ext_gru_500", help="the name of the models, to save the model")
    parser.add_argument('--bert_folder', type=str, default="hfl", help="The folder name that contains the BERT model")
    parser.add_argument('--bert_model_name', type=str, default="chinese-roberta-wwm-ext",
                        help="The bert model name to used")

    parser.add_argument('--height', type=int, default=10, help="the model height")
    parser.add_argument('--train_max_height', type=int, default = 14, help="the maximum height for training data")
    parser.add_argument('--consider_multiple_m0', type=int, default = 1, help="whether or not to consider multiple m0")

    parser.add_argument('--var_update_mode', type=str, default="gru", help="variable update mode")

    # training
    parser.add_argument('--mode', type=str, default="train", choices=["train", "test", "second_stage_train"], help="learning rate of the AdamW optimizer")
    parser.add_argument('--learning_rate', type=float, default=2e-5, help="learning rate of the AdamW optimizer")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="The maximum gradient norm")
    parser.add_argument('--num_epochs', type=int, default=550, help="The number of epochs to run")
    parser.add_argument('--fp16', type=int, default=0, choices=[0,1], help="using fp16 to train the model")

    parser.add_argument('--parallel', type=int, default=0, choices=[0,1], help="parallelizing model")

    # testing a pretrained model
    parser.add_argument('--cut_off', type=float, default=-100, help="cut off probability that we don't want to answer")
    parser.add_argument('--print_error', type=int, default=0, choices=[0, 1], help="whether to print the errors")
    parser.add_argument('--error_file', type=str, default="results/error.json", help="The file to print the errors")
    parser.add_argument('--result_file', type=str, default="results/res.json",
                        help="The file to print the errors")

    args = parser.parse_args()
    for k in args.__dict__:
        logger.info(f"{k} = {args.__dict__[k]}")
    return args


def train(config: Config, train_dataloader: DataLoader, num_epochs: int,
          bert_model_name: str, num_labels: int,
          dev: torch.device, tokenizer: PreTrainedTokenizer, valid_dataloader: DataLoader = None, test_dataloader: DataLoader = None,
          constant_values: List = None, res_file:str = None, error_file:str = None, output_lang =None, uni_labels = None):

    gradient_accumulation_steps = 1
    t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_epochs)

    constant_num = len(constant_values) if constant_values else 0

    MODEL_CLASS = UniversalModel
    model = MODEL_CLASS.from_pretrained(bert_model_name,
                                        num_labels=num_labels,
                                        height=config.height,
                                        constant_values=constant_values, uni_labels=uni_labels,
                                        add_replacement=bool(config.add_replacement),
                                        consider_multiple_m0=bool(config.consider_multiple_m0),
                                        var_update_mode=config.var_update_mode, return_dict=True,
                                        dev=dev, beam_size=4, max_tree_length=25, lang=output_lang).to(dev)



    if config.parallel:
        model = nn.DataParallel(model)

    scaler = None
    if config.fp16:
        scaler = torch.cuda.amp.GradScaler(enabled=bool(config.fp16))

    optimizer, scheduler = get_optimizers(config, model, t_total, weight_decay=1e-2)
    model.zero_grad()

    best_val_acc_performance = -1
    os.makedirs(f"model_files/{config.model_folder}", exist_ok=True)
    best_test_acc_performance = -1
    for epoch in range(num_epochs):
        total_loss = 0

        iter_loss = 0
        iter_tree_loss = 0
        iter_re_loss = 0
        iter_consistent_loss = 0


        model.train()
        model.tree_non_zero = torch.tensor(0)
        model.re_non_zero = torch.tensor(0)

        for iter, feature in tqdm(enumerate(train_dataloader, 1), desc="--training batch", total=len(train_dataloader)):
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=bool(config.fp16)):
                loss, loss_RE, loss_tree, consistent_loss  = \
                    model(dev, input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                             token_type_ids= feature.token_type_ids.to(dev),
                             variable_indexs_start= feature.variable_indexs_start.to(dev),
                             variable_indexs_end= feature.variable_indexs_end.to(dev),
                             num_variables = feature.num_variables.to(dev), num_list =feature.num_list,
                             variable_index_mask= feature.variable_index_mask.to(dev),
                             labels = feature.labels.to(dev), label_height_mask = feature.label_height_mask.to(dev),
                             target_len=feature.target_len, target_idx = feature.target_idx, RE_tree_align=feature.RE_tree_align,
                             seq_infix_idx = feature.seq_infix_idx, seq_infix_len = feature.seq_infix_len,
                          return_dict=True, updata_iter = epoch * len(train_dataloader) + iter).loss
            iter_tree_loss += loss_tree
            iter_re_loss += loss_RE
            iter_consistent_loss += consistent_loss
            iter_loss += loss

            if iter % 50 == 0:
                print('loss_RE', iter_re_loss.item()/50., '--', 'loss_tree', iter_tree_loss.item()/50, '--', 'consistent_loss', iter_consistent_loss.item()/50 )
                writer.add_scalar('training loss all', iter_loss.item()/50., global_step=(epoch) * len(train_dataloader) + iter)
                writer.add_scalar('training loss_RE', iter_re_loss.item()/50., global_step=(epoch ) * len(train_dataloader) + iter)
                writer.add_scalar('training loss_tree', iter_tree_loss.item()/50., global_step=(epoch ) * len(train_dataloader) + iter)
                writer.add_scalar(' consistent loss', iter_consistent_loss.item()/50., global_step=(epoch ) * len(train_dataloader) + iter)

                iter_loss, iter_tree_loss ,iter_re_loss ,iter_consistent_loss  = 0, 0, 0, 0

            if config.parallel:
                loss = loss.sum()
            if config.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            total_loss += loss.item()

            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)


            if config.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            model.zero_grad()
            if iter % 1000 == 0:
                logger.info(f"epoch: {epoch}, iteration: {iter}, current mean loss: {total_loss/iter:.2f}")


        logger.info(f"Finish epoch: {epoch}, loss: {total_loss:.2f}, mean loss: {total_loss/len(train_dataloader):.2f}")
        print('model.tree_non_zero, model.re_non_zero', model.tree_non_zero, model.re_non_zero)
        test_equ_acc, test_val_acc= -1, -1

        if test_dataloader is not None and epoch % 1 == 0:
            test_equ_acc, test_val_acc, tree_equ_acc, tree_val_acc, joint_val = evaluate(test_dataloader, model, dev, output_lang, uni_labels=config.uni_labels, fp16=bool(config.fp16),
                                                                              constant_values=constant_values,add_replacement=bool(config.add_replacement),
                                                                              consider_multiple_m0=bool(config.consider_multiple_m0),res_file=res_file,
                                                                              err_file=error_file, print_detail=1)

            writer.add_scalar('equ_acc', test_equ_acc, global_step=(epoch + 1) )
            writer.add_scalar('val_acc_performance', test_val_acc, global_step=(epoch + 1))
            writer.add_scalar('equ_tree_acc', tree_equ_acc, global_step=(epoch + 1))
            writer.add_scalar('val_tree_acc', tree_val_acc, global_step=(epoch + 1))
            writer.add_scalar('joint_val_acc', joint_val, global_step=(epoch + 1))

            print('epoch-epoch-epoch:',epoch,'val_RE_acc', test_val_acc,'val_tree_acc', tree_val_acc, 'joint_val', joint_val)

        if test_val_acc > best_test_acc_performance or tree_val_acc > best_test_acc_performance or joint_val >best_test_acc_performance:
            logger.info(f"[Model Info] Saving the best model with best valid val acc {test_val_acc:.6f} at epoch {epoch} ("
                        f"valid_equ: {test_equ_acc:.6f}, valid_val: {test_val_acc:.6f}")
            best_test_acc_performance = max(test_val_acc, tree_val_acc, joint_val)
            model_to_save = model.module if hasattr(model, "module") else model
            model_to_save.save_pretrained(f"model_files/{config.model_folder}")
            tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    logger.info(f"[Model Info] Best validation performance: {best_val_acc_performance}")

    model = MODEL_CLASS.from_pretrained(f"model_files/{config.model_folder}",
                                        num_labels=num_labels,
                                        height=config.height,
                                        constant_values=constant_values, uni_labels=uni_labels,
                                        add_replacement=bool(config.add_replacement),
                                        consider_multiple_m0=bool(config.consider_multiple_m0),
                                        var_update_mode=config.var_update_mode, return_dict=True, dev=dev, beam_size=4,
                                        max_tree_length=25).to(dev)


    if config.fp16:
        model.half()
        model.save_pretrained(f"model_files/{config.model_folder}")
        tokenizer.save_pretrained(f"model_files/{config.model_folder}")
    return model

def get_batched_prediction_consider_multiple_m0(feature, all_logits: torch.FloatTensor, constant_num: int, add_replacement: bool = False):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    device = feature.variable_indexs_start.device
    batched_prediction = [[] for _ in range(batch_size)]
    batch_score= [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        current_max_num_variable = max_num_variable + constant_num + k
        num_var_range = torch.arange(0, current_max_num_variable, device=feature.variable_indexs_start.device)
        combination = torch.combinations(num_var_range, r=2, with_replacement=add_replacement)
        num_combinations, _ = combination.size()
        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
        best_m0_score, best_comb = best_temp_score.max(dim=-1)
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label]

        best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                          best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(device)).squeeze(1)
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label, score) in enumerate(zip(best_comb_var_idxs, best_labels, curr_best_stop_labels,best_m0_score)):
            left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
            batch_score[b_idx].append(score.item())
    return batched_prediction, batch_score


def get_batched_prediction(feature, all_logits: torch.FloatTensor, constant_num: int, add_replacement: bool = False):
    batch_size, max_num_variable = feature.variable_indexs_start.size()
    max_num_variable = max_num_variable + constant_num
    num_var_range = torch.arange(0, max_num_variable, device=feature.variable_indexs_start.device)
    combination = torch.combinations(num_var_range, r=2, with_replacement=add_replacement)
    num_combinations, _ = combination.size()
    batched_prediction = [[] for _ in range(batch_size)]
    for k, logits in enumerate(all_logits):
        best_temp_logits, best_temp_stop_label = logits.max(dim=-1)
        best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
        best_m0_score, best_comb = best_temp_score.max(dim=-1)
        best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)
        b_idxs = [bidx for bidx in range(batch_size)]
        best_stop_label = best_temp_stop_label[b_idxs, best_comb, best_label]
        if k == 0:
            best_comb_var_idxs = torch.gather(combination.unsqueeze(0).expand(batch_size, num_combinations, 2), 1,
                                              best_comb.unsqueeze(1).unsqueeze(2).expand(batch_size, 1, 2).to(
                                                  feature.variable_indexs_start.device)).squeeze(1)
        else:
            best_comb_var_idxs = best_comb
        best_comb_var_idxs = best_comb_var_idxs.cpu().numpy()
        best_labels = best_label.cpu().numpy()
        curr_best_stop_labels = best_stop_label.cpu().numpy()
        for b_idx, (best_comb_idx, best_label, stop_label) in enumerate(
                zip(best_comb_var_idxs, best_labels, curr_best_stop_labels)):
            if isinstance(best_comb_idx, np.int64):
                right = best_comb_idx
                left = -1
            else:
                left, right = best_comb_idx
            curr_label = [left, right, best_label, stop_label]
            batched_prediction[b_idx].append(curr_label)
    return batched_prediction

def evaluate(valid_dataloader: DataLoader, model: nn.Module, dev: torch.device, output_lang ,fp16:bool, constant_values: List, uni_labels:List,
             add_replacement: bool = False, consider_multiple_m0: bool = False, res_file: str= None, err_file:str = None,
             num_beams:int = 1,print_detail = 1) -> Tuple[float, float,float, float, float]:
    model.eval()
    predictions = []
    score_re =[]
    labels = []
    labels_tree = []
    predictions_tree = []
    labels_tree_len = []

    constant_num = len(constant_values) if constant_values else 0
    with torch.no_grad():
        for index, feature in tqdm(enumerate(valid_dataloader), desc="--validation", total=len(valid_dataloader)):
            with torch.cuda.amp.autocast(enabled=fp16):
                module = model.module if hasattr(model, 'module') else model
                if num_beams == 1:
                    all_logits, tree_pred_batch, test_tree_score,_= module(dev, input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                                 token_type_ids=feature.token_type_ids.to(dev),
                                 variable_indexs_start=feature.variable_indexs_start.to(dev),
                                 variable_indexs_end=feature.variable_indexs_end.to(dev),
                                 num_variables = feature.num_variables.to(dev),
                                 variable_index_mask= feature.variable_index_mask.to(dev),
                                 labels=feature.labels.to(dev), label_height_mask= feature.label_height_mask.to(dev),
                                 return_dict=True, is_eval=True, target_len=feature.target_len, target_idx=feature.target_idx, num_list = feature.num_list).all_logits
                    batched_prediction, batched_score = get_batched_prediction(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement) \
                        if not consider_multiple_m0 else get_batched_prediction_consider_multiple_m0(feature=feature, all_logits=all_logits, constant_num=constant_num, add_replacement=add_replacement)
                else:
                    batched_prediction, _ =  module.beam_search(input_ids=feature.input_ids.to(dev), attention_mask=feature.attention_mask.to(dev),
                                 token_type_ids=feature.token_type_ids.to(dev),
                                 variable_indexs_start=feature.variable_indexs_start.to(dev),
                                 variable_indexs_end=feature.variable_indexs_end.to(dev),
                                 num_variables = feature.num_variables.to(dev),
                                 variable_index_mask= feature.variable_index_mask.to(dev),
                                 labels=feature.labels.to(dev), label_height_mask= feature.label_height_mask.to(dev),
                                 return_dict=True, is_eval=True, num_beams=num_beams)
                    batched_prediction = batched_prediction[:, 0, :, :].numpy().astype(int).tolist()

                accum_score = [0 for _ in range(feature.variable_indexs_start.size(0))]
                for b, (inst_predictions ,score) in enumerate(zip(batched_prediction, batched_score)):
                    for p, (prediction_step, score_layer) in enumerate(zip(inst_predictions, score)):
                        left, right, op_id, stop_id = prediction_step
                        accum_score[b] = accum_score[b] + score_layer
                        if stop_id == 1:
                            batched_prediction[b] = batched_prediction[b][:(p+1)]
                            break
                batched_labels = feature.labels.cpu().numpy().tolist()
                batched_labels_tree = feature.target_idx.cpu().numpy().tolist()
                batched_labels_len = feature.target_len.cpu().numpy().tolist()

                for b, inst_labels in enumerate(batched_labels):
                    for p, label_step in enumerate(inst_labels):
                        left, right, op_id, stop_id = label_step
                        if stop_id == 1:
                            batched_labels[b] = batched_labels[b][:(p+1)]
                            break
                predictions.extend(batched_prediction)
                score_re.extend(accum_score)
                labels.extend(batched_labels)
                labels_tree.extend(batched_labels_tree)
                predictions_tree.extend(tree_pred_batch)
                labels_tree_len.extend(batched_labels_len)

        print('model.tree_non_zero, model.re_non_zero', model.tree_non_zero, model.re_non_zero)

    acc, val_acc, tree_equ_acc, tree_val_acc, joint_val_acc = cal_acc(valid_dataloader, predictions, labels, predictions_tree,
                                                                      labels_tree, labels_tree_len, output_lang,
                                                                      print_detail=0, uni_labels=uni_labels, constant_num=constant_num,
                                                                      constant_values=constant_values, consider_multiple_m0=1,
                                                                      res_file=res_file, err_file=err_file)



    return acc, val_acc, tree_equ_acc, tree_val_acc, joint_val_acc


def cal_acc(valid_dataloader,predictions,labels,predictions_tree, labels_tree, labels_tree_len ,output_lang, select_all=None,
            print_detail=0,uni_labels=None, constant_num=None,constant_values=None,consider_multiple_m0=1,res_file=None,err_file=None):
    corr = 0
    num_label_step_corr = Counter()
    num_label_step_total = Counter()
    insts = valid_dataloader.dataset.insts
    number_instances_remove = valid_dataloader.dataset.number_instances_remove
    res_re_equ = []
    for inst_predictions, inst_labels in zip(predictions, labels):
        num_label_step_total[len(inst_labels)] += 1
        if len(inst_predictions) != len(inst_labels):
            res_re_equ.append(False)
            continue
        is_correct = True
        for prediction_step, label_step in zip(inst_predictions, inst_labels):
            if prediction_step != label_step:
                is_correct = False
                break
        if is_correct:
            num_label_step_corr[len(inst_labels)] += 1
            corr += 1
            res_re_equ.append(True)

        else:
            res_re_equ.append(False)

    total = len(labels)
    adjusted_total = total + number_instances_remove
    acc = corr * 1.0 / adjusted_total
    if print_detail:
        logger.info(
            f"[Info] Equation accuracy: {acc * 100:.2f}%, total: {total}, corr: {corr}, adjusted_total: {adjusted_total}")

    ## value accuarcy
    val_corr = 0
    num_label_step_val_corr = Counter()
    err = []
    corr += 0
    joint_val = []

    res_re_val = []
    for inst_predictions, inst_labels, inst in zip(predictions, labels, insts):
        num_list = inst["num_list"]
        is_value_corr, predict_value, gold_value, pred_ground_equation, gold_ground_equation = is_value_correct(
            inst_predictions, inst_labels, num_list, num_constant=constant_num, uni_labels=uni_labels,
            constant_values=constant_values, consider_multiple_m0=consider_multiple_m0)

        val_corr += 1 if is_value_corr else 0
        if is_value_corr:
            num_label_step_val_corr[len(inst_labels)] += 1
            corr += 1
            res_re_val.append(True)


        else:
            err.append(inst)
            res_re_val.append(False)
        inst["predict_value"] = predict_value
        inst["gold_value"] = gold_value
        inst['pred_ground_equation'] = pred_ground_equation
        inst['gold_ground_equation'] = gold_ground_equation

    val_acc = val_corr * 1.0 / adjusted_total

    if print_detail == 1:

        logger.info(
            f"[Info] Value accuracy: {val_acc * 100:.2f}%, total: {total}, corr: {corr}, adjusted_total: {adjusted_total}")

        for key in num_label_step_total:
            curr_corr = num_label_step_corr[key]
            curr_val_corr = num_label_step_val_corr[key]
            curr_total = num_label_step_total[key]
            logger.info(
                f"[Info] step num: {key} Acc.:{curr_corr * 1.0 / curr_total * 100:.2f} ({curr_corr}/{curr_total}) val acc: {curr_val_corr * 1.0 / curr_total * 100:.2f} ({curr_val_corr}/{curr_total})")
        if res_file is not None:
            write_data(file=res_file, data=insts)
        if err_file is not None:
            write_data(file=err_file, data=err)

    tree_val_corr = 0
    tree_equ_corr = 0
    res_tree_equ = []
    res_tree_val = []
    for inst_predictions, inst_labels, inst_len_tree, inst in zip(predictions_tree, labels_tree, labels_tree_len,
                                                                  insts):
        num_list = inst["num_list"]
        # print(inst_predictions,inst_labels[:inst_len_tree])
        val_ac, equ_ac, test, tar = compute_prefix_tree_result(inst_predictions, inst_labels[:inst_len_tree],
                                                               output_lang, num_list)
        if val_ac:
            tree_val_corr += 1
            res_tree_val.append(True)


        else:
            res_tree_val.append(False)

        if equ_ac:
            tree_equ_corr += 1
            res_tree_equ.append(True)
        else:
            res_tree_equ.append(False)
        # print(val_ac, equ_ac, test, tar)

    tree_val_acc = tree_val_corr * 1.0 / adjusted_total
    tree_equ_acc = tree_equ_corr * 1.0 / adjusted_total

    joint_val_res = 0
    if select_all is not None:
        for re, tree, flag in zip(res_re_val, res_tree_val, select_all):
            if flag == 0:
                joint_val.append(re)
            else:
                joint_val.append(tree)
        for joint in joint_val:
            if joint:
                joint_val_res += 1
    print('joint:', joint_val_res)
    joint_val_acc = joint_val_res * 1.0 / adjusted_total


    return acc, val_acc, tree_equ_acc, tree_val_acc, joint_val_acc




def main():
    parser = argparse.ArgumentParser(description="classificaton")
    opt = parse_arguments(parser)
    set_seed(opt.seed)
    conf = Config(opt)
    torch.cuda.set_device(int(opt.device[-1]))
    bert_model_name = conf.bert_model_name if conf.bert_folder == "" or conf.bert_folder=="none" else f"{conf.bert_folder}/{conf.bert_model_name}"
    class_name_2_tokenizer = {
        "bert-base-cased": BertTokenizerFast,
        "roberta-base": RobertaTokenizerFast,
        "bert-base-multilingual-cased": BertTokenizerFast,
        "xlm-roberta-base": XLMRobertaTokenizerFast,
        'bert-base-chinese': BertTokenizerFast,
        'hfl/chinese-bert-wwm-ext': BertTokenizerFast,
        'hfl/chinese-roberta-wwm-ext': BertTokenizerFast,
    }

    TOKENIZER_CLASS_NAME = class_name_2_tokenizer[bert_model_name]

    tokenizer = TOKENIZER_CLASS_NAME.from_pretrained(bert_model_name)


    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    num_labels = len(uni_labels)
    conf.uni_labels = uni_labels
    if conf.use_constant:
        if "23k" in conf.train_file:
            constant2id = {"1": 0, "PI": 1}
            constant_values = [1.0, 3.14]
            constant_number = len(constant_values)
        else:
            constant2id = None
            constant_values = None
            constant_number = 0
    else:
        raise NotImplementedError
    logger.info(f"[Data Info] constant info: {constant2id}")


    if opt.mode == "train":
        logger.info("[Data Info] Reading training data")
        dataset = UniversalDataset(file=conf.train_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.train_num, filtered_steps=opt.train_filtered_steps,
                                   constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0),
                                   data_max_height=opt.train_max_height, pretrained_model_name=bert_model_name)
        output_lang = dataset.output_lang
        logger.info("[Data Info] Reading validation data")
        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, uni_labels=conf.uni_labels, number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                                   use_incremental_labeling=bool(conf.consider_multiple_m0),
                                        data_max_height=conf.height, pretrained_model_name=bert_model_name)

        logger.info("[Data Info] Reading Testing data data")
        test_dataset = None
        if os.path.exists(conf.test_file):
            test_dataset = UniversalDataset(file=conf.test_file, tokenizer=tokenizer, uni_labels=conf.uni_labels,
                                            number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                            constant2id=constant2id, constant_values=constant_values,
                                            add_replacement=bool(conf.add_replacement),
                                            use_incremental_labeling=bool(conf.consider_multiple_m0),
                                            data_max_height=conf.height, pretrained_model_name=bert_model_name)
        logger.info(f"[Data Info] Training instances: {len(dataset)}, Validation instances: {len(eval_dataset)}")
        if test_dataset is not None:
            logger.info(f"[Data Info] Testing instances: {len(test_dataset)}")
        logger.info("[Data Info] Loading data")
        train_dataloader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=conf.num_workers, collate_fn=dataset.collate_function)
        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)
        test_loader = None
        if test_dataset is not None:
            logger.info("[Data Info] Loading Test data")
            test_loader = DataLoader(test_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=conf.num_workers, collate_fn=eval_dataset.collate_function)

        res_file = f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        # Train the model
        model = train(conf, train_dataloader,
                      num_epochs= conf.num_epochs,
                      bert_model_name = bert_model_name,
                      valid_dataloader = valid_dataloader, test_dataloader=test_loader,
                      dev=conf.device, tokenizer=tokenizer, num_labels=num_labels,
                      constant_values=constant_values, res_file=res_file, error_file=err_file,output_lang=output_lang,uni_labels = conf.uni_labels)


        test_equ_acc, test_val_acc, tree_equ_acc, tree_val_acc, joint_val = evaluate(valid_dataloader, model, conf.device, output_lang, fp16=bool(conf.fp16), constant_values=constant_values,
                 add_replacement=bool(conf.add_replacement), consider_multiple_m0=bool(conf.consider_multiple_m0), uni_labels=conf.uni_labels)

    else:
        logger.info(f"Testing the model now.")

        MODEL_CLASS = UniversalModel

        eval_dataset = UniversalDataset(file=conf.dev_file, tokenizer=tokenizer, uni_labels=conf.uni_labels,
                                        number=conf.dev_num, filtered_steps=opt.test_filtered_steps,
                                        constant2id=constant2id, constant_values=constant_values,
                                        add_replacement=bool(conf.add_replacement),
                                        use_incremental_labeling=bool(conf.consider_multiple_m0),
                                        data_max_height=conf.height, pretrained_model_name=bert_model_name)

        output_lang = eval_dataset.output_lang
        model = MODEL_CLASS.from_pretrained(f"model_files/{conf.model_folder}",
                                               num_labels=num_labels,
                                               height = conf.height,
                                            constant_values=constant_values, uni_labels=uni_labels,
                                            add_replacement=bool(conf.add_replacement), consider_multiple_m0=conf.consider_multiple_m0,
                                            var_update_mode=conf.var_update_mode,beam_size=4, max_tree_length=25, dev = conf.device, lang=output_lang).to(conf.device)






        logger.info("[Data Info] Reading test data")

        valid_dataloader = DataLoader(eval_dataset, batch_size=conf.batch_size, shuffle=False, num_workers=0,
                                      collate_fn=eval_dataset.collate_function)
        output_lang = eval_dataset.output_lang
        os.makedirs("results", exist_ok=True)
        res_file= f"results/{conf.model_folder}.res.json"
        err_file = f"results/{conf.model_folder}.err.json"
        test_equ_acc, test_val_acc, tree_equ_acc, tree_val_acc,joint_val = evaluate(valid_dataloader, model, conf.device, output_lang, uni_labels=conf.uni_labels, fp16=bool(conf.fp16), constant_values=constant_values, add_replacement=bool(conf.add_replacement),
                 consider_multiple_m0=bool(conf.consider_multiple_m0), res_file=res_file, err_file=err_file)


        print(test_equ_acc, test_val_acc, tree_equ_acc, tree_val_acc)




if __name__ == "__main__":
    main()

