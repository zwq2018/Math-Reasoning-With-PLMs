import copy

from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel, BertConfig
import torch.nn as nn
import torch
import torch.utils.checkpoint
import torch.autograd as autograd
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    ModelOutput,
)
import copy
import random
from dataclasses import dataclass
from typing import Optional, List
from src.model.tree_decoder import GenerateNode, Merge, Prediction
from src.model.tree_decoder import tree_decoder, evaluate_tree_batch
from src.eval.utils import is_value_correct, compute_prefix_tree_result, compute_prefix_expression
@dataclass
class UniversalOutput(ModelOutput):
    """
    Base class for ß of sentence classification models.

    Args:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`labels` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
    """

    loss: Optional[torch.FloatTensor] = None
    all_logits: List[torch.FloatTensor] = None

def get_combination_mask(batched_num_variables: torch.Tensor, combination: torch.Tensor):
    """

    """
    batch_size, = batched_num_variables.size()       # batch-size
    num_combinations, _ = combination.size()
    batched_num_variables = batched_num_variables.unsqueeze(1).unsqueeze(2).expand(batch_size, num_combinations, 2)
    batched_combination = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
    batched_comb_mask = torch.lt(batched_combination, batched_num_variables)
    return batched_comb_mask[:,:, 0] * batched_comb_mask[:,:, 1]


class UniversalModel(BertPreTrainedModel):

    def __init__(self, config: BertConfig,
                 height: int = 4,
                 constant_values=None, uni_labels = None,
                 add_replacement: bool = False,
                 consider_multiple_m0: bool = False, var_update_mode: str= 'gru',dev=None, beam_size=8, max_tree_length = 45, lang=None):
        """
        Constructor for model function
        :param config:
        :param diff_param_for_height: whether we want to use different layers/parameters for different height
        :param height: the maximum number of height we want to use
        :param constant_num: the number of constant we consider
        :param add_replacement: only at h=0, whether we want to consider somehting like "a*a" or "a+a"
                                also applies to h>0 when `consider_multplie_m0` = True
        :param consider_multiple_m0: considering more m0 in one single step. for example soemthing like "m3 = m1 x m2".
        """
        super().__init__(config)
        self.num_labels = config.num_labels ## should be 6
        assert self.num_labels == 6
        self.config = config

        self.bert = BertModel(config)
        self.add_replacement = bool(add_replacement)
        self.consider_multiple_m0 = bool(consider_multiple_m0)

        self.label_rep2label = nn.Linear(config.hidden_size, 1)
        self.max_height = height
        # self.linears = nn.ModuleList()
        # for i in range(self.num_labels):
        #     self.linears.append(nn.Sequential(
        #         nn.Linear(3 * config.hidden_size, config.hidden_size),
        #         nn.ReLU(),
        #         nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
        #         nn.Dropout(config.hidden_dropout_prob)

        self.merge_op_num = nn.Sequential(nn.Linear(3 * config.hidden_size, config.hidden_size),
                                          nn.ReLU(),
                                          nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                                          nn.Dropout(config.hidden_dropout_prob) )

        self.constant_values = [str(x) for x in constant_values]
        self.uni_labels = uni_labels

        self.stopper_transformation = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                    nn.Dropout(config.hidden_dropout_prob)
                )

        self.stopper = nn.Linear(config.hidden_size, 2)
        self.variable_gru = None
        if var_update_mode == 'gru':
            self.var_update_mode = 0
        elif var_update_mode == 'attn':
            self.var_update_mode = 1
        else:
            self.var_update_mode = -1
        if self.consider_multiple_m0:
            if var_update_mode == 'gru':
                self.variable_gru = nn.GRUCell(config.hidden_size, config.hidden_size)
            elif var_update_mode == 'attn':
                self.variable_gru = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=6, batch_first=True)
            else:
                print("[WARNING] no rationalizer????????")
                self.variable_gru = None
        self.constant_num = len(self.constant_values)
        self.constant_emb = None
        if self.constant_num > 0:
            self.const_rep = nn.Parameter(torch.randn(self.constant_num, config.hidden_size))

        self.op_embedding = nn.Parameter(torch.randn(self.num_labels, config.hidden_size))
        self.merge_hidden = nn.Linear(config.hidden_size * 3, config.hidden_size)
        self.lang = lang


        self.variable_scorer = nn.Sequential(
                    nn.Linear(config.hidden_size, config.hidden_size),
                    nn.ReLU(),
                    nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps),
                    nn.Dropout(config.hidden_dropout_prob),
                    nn.Linear(config.hidden_size, 1),
                )
        self.gru_pade = nn.GRU(config.hidden_size,  config.hidden_size, num_layers=1, bidirectional=True, batch_first=True, device=dev)
        self.encoder_merge = nn.Linear(in_features=config.hidden_size * 1, out_features=config.hidden_size)

        self.predict_model = Prediction(hidden_size=config.hidden_size, op_nums=self.num_labels,input_size=self.constant_num)
        self.generate_model =  GenerateNode(hidden_size=config.hidden_size, op_nums=self.num_labels)
        self.merge_model = Merge(hidden_size=config.hidden_size, embedding_size=config.hidden_size)
        #
        self.generate_model.to(dev)
        self.merge_model.to(dev)
        self.predict_model.to(dev)
        self.beam = beam_size
        self.max_length = max_tree_length
        self.project_align = nn.Linear(config.hidden_size, config.hidden_size)
        self.updata_num = 0
        self.tree_non_zero = torch.tensor(0)
        self.re_non_zero = torch.tensor(0)

        self.init_weights()


    def forward(self,dev: torch.device,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        variable_indexs_start: torch.Tensor = None,
        variable_indexs_end: torch.Tensor = None,
        num_variables: torch.Tensor = None, # batch_size [3,4]
        num_list = None,
        variable_index_mask:torch.Tensor = None, # batch_size x num_variable
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        label_height_mask = None,
        output_attentions=None,
        output_hidden_states=None,
        target_len = None,
        target_idx = None,
        RE_tree_align =None,
        seq_infix_idx=None,
        seq_infix_len =None,
        return_dict=None,
        is_eval=False,
        updata_iter = 0 ):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert( # batch_size, sent_len, hidden_size,
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        batch_size, sent_len, hidden_size = outputs.last_hidden_state.size()

        padding_hidden = torch.zeros([2, batch_size, hidden_size],device=dev)
        outs, hiddens = self.gru_pade(outputs.last_hidden_state.to(dev), padding_hidden)

        problem_out = outs[:, -1, :hidden_size] + outs[:, 0, hidden_size:]

        if labels is not None and not is_eval:
            _, max_height, _ = labels.size()
        else:
            max_height = self.max_height

        _, max_num_variable = variable_indexs_start.size()
        var_sum = (variable_indexs_start - variable_indexs_end).sum()
        var_start_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_start.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
        if var_sum != 0:
            var_end_hidden_states = torch.gather(outputs.last_hidden_state, 1, variable_indexs_end.unsqueeze(-1).expand(batch_size, max_num_variable, hidden_size))
            var_hidden_states = var_start_hidden_states + var_end_hidden_states
            num_embedding = var_start_hidden_states + var_end_hidden_states
        else:
            var_hidden_states = var_start_hidden_states
        if self.constant_num > 0:
            constant_hidden_states = self.const_rep.unsqueeze(0).expand(batch_size, self.constant_num, hidden_size)
            var_hidden_states = torch.cat([constant_hidden_states, var_hidden_states], dim=1)
            num_variables = num_variables + self.constant_num
            max_num_variable = max_num_variable + self.constant_num
            const_idx_mask = torch.ones((batch_size, self.constant_num), device=variable_indexs_start.device)
            variable_index_mask = torch.cat([const_idx_mask, variable_index_mask], dim = 1)

        best_mi_label_rep = None
        loss = 0
        all_logits = []
        best_mi_scores = None
        batch_rep = [[] for _ in range(batch_size)]
        batch_rep_pred = [[] for _ in range(batch_size)]
        for i in range(max_height):
            if i == 0:
                num_var_range = torch.arange(0, max_num_variable, device=variable_indexs_start.device)
                combination = torch.combinations(num_var_range, r=2, with_replacement=self.add_replacement)
                num_combinations, _ = combination.size()  # number_of_combinations x 2
                batched_combination_mask = get_combination_mask(batched_num_variables=num_variables, combination=combination)  # batch_size, num_combinations
                var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                m0_hidden_states = torch.cat([expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]], dim=-1)  #


                merge_m0_hidden_states = torch.tanh(self.merge_hidden(m0_hidden_states))
                stack_m0_label_rep_op = []
                for j in range(self.num_labels):
                    op_embedding_expand = self.op_embedding[j, :].unsqueeze(0).unsqueeze(0).expand(batch_size,num_combinations,hidden_size)  # (batch, num_com, hidden_size)
                    temp = self.merge_op_num(torch.cat([merge_m0_hidden_states, op_embedding_expand, merge_m0_hidden_states * op_embedding_expand],dim=-1))
                    stack_m0_label_rep_op.append(temp)
                m0_label_rep_op = torch.stack(stack_m0_label_rep_op, dim=2)  # (batah, num_com, 6, hidden-size)


                m0_logits = self.label_rep2label(m0_label_rep_op).expand(batch_size, num_combinations, self.num_labels, 2)
                m0_logits = m0_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).float().log()
                m0_stopper_logits = self.stopper(self.stopper_transformation(m0_label_rep_op))

                var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)
                expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size, -1)).unsqueeze(-1).view(batch_size, num_combinations, 2)
                expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2)

                m0_combined_logits = m0_logits + m0_stopper_logits

                all_logits.append(m0_combined_logits)
                best_temp_logits, best_stop_label =  m0_combined_logits.max(dim=-1)
                best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
                best_m0_score, best_comb = best_temp_score.max(dim=-1)
                best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)

                b_idxs = [k for k in range(batch_size)]

                if labels is not None and not is_eval:
                    m0_gold_labels = labels[:, i, :]
                    m0_gold_comb = m0_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                    batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                    judge = m0_gold_comb == batched_comb
                    judge = judge[:, :, 0] * judge[:, :, 1]
                    judge = judge.nonzero()[:,1]
                    m0_gold_scores = m0_combined_logits[b_idxs, judge, m0_gold_labels[:, 2], m0_gold_labels[:, 3]]
                    loss = loss + (best_m0_score - m0_gold_scores).sum()

                    best_mi_label_rep = m0_label_rep_op[b_idxs, judge, m0_gold_labels[:, 2]]
                    best_mi_scores = m0_logits[b_idxs, judge, m0_gold_labels[:, 2]][:, 0]

                    for bidx, rep_single in enumerate(best_mi_label_rep):
                         rep_single = self.project_align(rep_single.unsqueeze(0))
                         batch_rep_pred[bidx].append(rep_single)



                else:
                    best_m0_label_rep = m0_label_rep_op[b_idxs, best_comb, best_label]
                    best_mi_label_rep = best_m0_label_rep
                    best_mi_scores = m0_logits[b_idxs, best_comb, best_label][:, 0]


            else:
                if not self.consider_multiple_m0:
                    expanded_best_mi_label_rep = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable, hidden_size)
                    mi_sum_states = torch.cat([expanded_best_mi_label_rep, var_hidden_states, expanded_best_mi_label_rep * var_hidden_states], dim= -1)
                    mi_label_rep = torch.stack([layer(mi_sum_states) for layer in linear_modules], dim=2)

                    mi_logits = self.label_rep2label(mi_label_rep).expand(batch_size, max_num_variable, self.num_labels, 2)
                    mi_logits = mi_logits + variable_index_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, max_num_variable, self.num_labels, 2).float().log()


                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep))

                    mi_combined_logits = mi_logits + mi_stopper_logits

                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max( dim=-1)
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
                    best_m0_score, best_comb = best_temp_score.max(dim=-1)
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)

                    b_idxs = [k for k in range(batch_size)]
                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i, -3:]
                        height_mask = label_height_mask[:, i]
                        mi_gold_scores = mi_combined_logits[b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1], mi_gold_labels[:, 2]]
                        current_loss = (best_m0_score - mi_gold_scores) * height_mask
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep[b_idxs, mi_gold_labels[:, 0], mi_gold_labels[:, 1]]
                    else:
                        best_mi_label_rep = mi_label_rep[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                else:


                    if self.var_update_mode == 0:
                        init_h = best_mi_label_rep.unsqueeze(1).expand(batch_size, max_num_variable + i - 1, hidden_size).contiguous().view(-1, hidden_size)
                        gru_inputs = var_hidden_states.view(-1, hidden_size)
                        var_hidden_states = self.variable_gru(gru_inputs, init_h).view(batch_size, max_num_variable + i - 1, hidden_size)
                    elif self.var_update_mode == 1:
                        temp_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)
                        temp_mask = torch.eye(max_num_variable + i, device=variable_indexs_start.device)
                        temp_mask[:, 0] = 1
                        temp_mask[0, :] = 1
                        updated_all_states, _ = self.variable_gru(temp_states, temp_states, temp_states, attn_mask=1 - temp_mask)
                        var_hidden_states = updated_all_states[:, 1:, :]

                    num_var_range = torch.arange(0, max_num_variable + i, device=variable_indexs_start.device)
                    combination = torch.combinations(num_var_range, r=2,  with_replacement=self.add_replacement)
                    num_combinations, _ = combination.size()  # number_of_combinations x 2
                    batched_combination_mask = get_combination_mask(batched_num_variables=num_variables + i, combination=combination)

                    var_hidden_states = torch.cat([best_mi_label_rep.unsqueeze(1), var_hidden_states], dim=1)
                    var_comb_hidden_states = torch.gather(var_hidden_states, 1, combination.view(-1).unsqueeze(0).unsqueeze(-1).expand(batch_size, num_combinations * 2, hidden_size))
                    expanded_var_comb_hidden_states = var_comb_hidden_states.unsqueeze(-2).view(batch_size, num_combinations, 2, hidden_size)
                    mi_hidden_states = torch.cat( [expanded_var_comb_hidden_states[:, :, 0, :], expanded_var_comb_hidden_states[:, :, 1, :],expanded_var_comb_hidden_states[:, :, 0, :] * expanded_var_comb_hidden_states[:, :, 1, :]
                                            ], dim=-1)



                    merge_m1_hidden_states = torch.tanh(self.merge_hidden(mi_hidden_states))
                    stack_mi_label_rep_op=[]
                    for j in range(self.num_labels):
                        op_embedding_expand = self.op_embedding[j, :].unsqueeze(0).unsqueeze(0).expand(batch_size,num_combinations,hidden_size)
                        temp = self.merge_op_num(torch.cat([merge_m1_hidden_states, op_embedding_expand, merge_m1_hidden_states * op_embedding_expand],dim=-1))
                        stack_mi_label_rep_op.append(temp)
                    mi_label_rep_op = torch.stack(stack_mi_label_rep_op, dim=2)   # (batah, num_com, 6, hidden-size)


                    mi_logits = self.label_rep2label(mi_label_rep_op).expand(batch_size, num_combinations, self.num_labels, 2)
                    mi_logits = mi_logits + batched_combination_mask.unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels, 2).float().log()


                    mi_stopper_logits = self.stopper(self.stopper_transformation(mi_label_rep_op))
                    var_scores = self.variable_scorer(var_hidden_states).squeeze(-1)
                    expanded_var_scores = torch.gather(var_scores, 1, combination.unsqueeze(0).expand(batch_size, num_combinations, 2).view(batch_size,-1)).unsqueeze(-1).view(batch_size, num_combinations, 2)
                    expanded_var_scores = expanded_var_scores.sum(dim=-1).unsqueeze(-1).unsqueeze(-1).expand(batch_size, num_combinations, self.num_labels,  2)
                    mi_combined_logits = mi_logits + mi_stopper_logits

                    all_logits.append(mi_combined_logits)
                    best_temp_logits, best_stop_label = mi_combined_logits.max( dim=-1)
                    best_temp_score, best_temp_label = best_temp_logits.max(dim=-1)
                    best_mi_score, best_comb = best_temp_score.max(dim=-1)
                    best_label = torch.gather(best_temp_label, 1, best_comb.unsqueeze(-1)).squeeze(-1)  #


                    if labels is not None and not is_eval:
                        mi_gold_labels = labels[:, i, :]
                        mi_gold_comb = mi_gold_labels[:, :2].unsqueeze(1).expand(batch_size, num_combinations, 2)
                        batched_comb = combination.unsqueeze(0).expand(batch_size, num_combinations, 2)
                        judge = mi_gold_comb == batched_comb
                        judge = judge[:, :, 0] * judge[:, :, 1]
                        judge = judge.nonzero()[:, 1]
                        mi_gold_scores = mi_combined_logits[b_idxs, judge, mi_gold_labels[:, 2], mi_gold_labels[:, 3]]
                        height_mask = label_height_mask[:, i]
                        current_loss = (best_mi_score - mi_gold_scores) * height_mask
                        loss = loss + current_loss.sum()
                        best_mi_label_rep = mi_label_rep_op[b_idxs, judge, mi_gold_labels[:, 2]]
                        best_mi_scores = mi_logits[b_idxs, judge, mi_gold_labels[:, 2]][:, 0]

                        for bidx,(rep_single, mask_single) in enumerate(zip(best_mi_label_rep, height_mask)):
                            if mask_single :
                                rep_single = self.project_align(rep_single.unsqueeze(0))
                                batch_rep_pred[bidx].append(rep_single)


                    else:
                        best_mi_label_rep = mi_label_rep_op[b_idxs, best_comb, best_label]  # batch_size x hidden_size
                        best_mi_scores = mi_logits[b_idxs, best_comb, best_label][:, 0]


        encoder_p = self.encoder_merge(problem_out)
        if not is_eval :

            re_align = [RE_align_sinlge[:target_len[idx]].cpu().numpy().tolist() for idx, RE_align_sinlge in enumerate(RE_tree_align)]

            loss_tree, consistent_loss  = tree_decoder(dev, self.predict_model, self.generate_model, self.merge_model, encoder_p,
                                     num_embedding, self.const_rep, self.op_embedding, outputs.last_hidden_state,
                                     target_len, target_idx , attention_mask, variable_index_mask, batch_rep_pred, re_align)    #

            embeds = (num_embedding.cpu().detach().numpy(), self.const_rep.cpu().detach().numpy(),
                      self.op_embedding.cpu().detach().numpy(), outputs.last_hidden_state.cpu().detach().numpy())


            test_res = None
            loss_all = loss_tree + loss + consistent_loss
            test_score = torch.tensor(0.)

        else:
            loss_tree = torch.tensor(0.)
            consistent_loss = torch.tensor(0.)


            loss_all = loss_tree + loss + consistent_loss


            test_res, test_score, tree_global_rep = evaluate_tree_batch(dev, self.predict_model, self.generate_model, self.merge_model, encoder_p, num_embedding,
                                     self.const_rep, self.op_embedding, outputs.last_hidden_state , attention_mask, variable_index_mask,
                                     beam_size=self.beam, max_length=self.max_length)



            embeds = (num_embedding.cpu().detach().numpy(), self.const_rep.cpu().detach().numpy(),
                  self.op_embedding.cpu().detach().numpy(), outputs.last_hidden_state.cpu().detach().numpy())
        return UniversalOutput(loss = (loss_all, loss, loss_tree, consistent_loss) , all_logits=(all_logits, test_res, test_score, embeds))




