import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
from torch.nn import functional
import re

mseloss = torch.nn.MSELoss(reduction='mean')
class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding        #
        self.left_flag = left_flag



class Prediction(nn.Module):

    def __init__(self, hidden_size, op_nums, input_size ,dropout=0.5):
        super(Prediction, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        self.dropout = nn.Dropout(dropout)

        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)
        self.leaf = nn.Linear(2 * hidden_size, hidden_size)
        self.ops = nn.Linear(hidden_size * 3, 1)

        self.attn = TreeAttn(hidden_size, hidden_size)

        self.score_num = Score(hidden_size)
        self.score_op = Score(hidden_size)
        self.project_align = nn.Linear(hidden_size,hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask=None, mask_nums=None, const_embed=None, op_embed=None):
        embedding_const = const_embed.unsqueeze(0)
        embedding_op = op_embed.unsqueeze(0)

        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]  #
                current_embeddings.append(current_node.embedding)
        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)
        seq_mask_bool_not = ~seq_mask.bool()                 #
        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs.transpose(0, 1), seq_mask_bool_not )
        current_context = current_attn.bmm(encoder_outputs)

        batch_size = current_embeddings.size(0)

        repeat_dims = [1] * embedding_const.dim()                   # [1, 1, 1]
        repeat_dims[0] = batch_size                                 # [batch, 1, 1]
        embedding_const = embedding_const.repeat(*repeat_dims)
        embedding_const_num = torch.cat((embedding_const, num_pades),dim=1)


        leaf_input = torch.cat((current_node, current_context), 2)        # (batch,1 ,hidden*2)
        leaf_input = self.dropout(leaf_input)                             # (batch, 1,hidden*2)
        leaf_input_merge = torch.tanh(self.leaf(leaf_input))              # (batch,1 ,hidden)




        embedding_const_num_ = self.dropout(embedding_const_num)
        mask_nums_bool_not = ~mask_nums.bool()
        num_score = self.score_num(leaf_input_merge, embedding_const_num_, mask_nums_bool_not)

        embedding_op_expand = embedding_op.repeat(*repeat_dims)
        embedding_op_expand_= self.dropout(embedding_op_expand)

        op_score = self.score_op(leaf_input_merge, embedding_op_expand_, None)



        return num_score, op_score, current_node, current_context, embedding_const_num, embedding_op.squeeze(0)



class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)
        repeat_dims = [1] * hidden.dim()       # [1,1,1]
        repeat_dims[0] = max_len               # [max_len,1,1]
        hidden = hidden.repeat(*repeat_dims)
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask.bool(), -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)

        return attn_energies.unsqueeze(1)




class Score(nn.Module):
    def __init__(self, hidden_size):
        super(Score, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(3 * hidden_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()      #
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings, hidden * num_embeddings), 2).view(-1, 3 * self.hidden_size)

        score = self.score(torch.tanh(self.attn(energy_in)))
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # ( batch,  num-size+2 )
        if num_mask is not None:
            score = score.masked_fill_(num_mask.bool(), -1e12)
        return score


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False, layer_align=-1):
        self.embedding = embedding
        self.terminal = terminal
        self.align = layer_align

class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + hidden_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context, op_embed=None ):
        node_label_op_embedding = op_embed[node_label]
        node_label_op = self.em_dropout(node_label_op_embedding)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label_op), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label_op), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label_op), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label_op), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_op_embedding


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree



def sequence_mask(sequence_length, dev=None, max_len=None):
    if max_len is None:
        max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len).to(dev)
    seq_length_expand = (sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand



def masked_cross_entropy(logits, target, length, dev):

    length = torch.LongTensor(length).to(dev)

    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    logits_flat = logits.view(-1, logits.size(-1))
    log_probs_flat = functional.log_softmax(logits_flat, dim=1)

    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)

    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, dev=dev, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum() / length.float().sum()

    return loss


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = deepcopy(out)





def tree_decoder(dev, predict, generate, merge, encoder_p, num_embed, const_embed, op_embed, text_seq_encoder,
                 target_len, target_idx, seq_mask, num_mask, batch_RE_rep, RE_tree_align):

    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    batch_size, hidden_size = encoder_p.size()
    node_stacks = [[TreeNode(hidden)] for hidden in encoder_p.split(1, dim=0)]
    left_childs = [None for _ in range(batch_size)]
    max_target_length = max(target_len)
    all_node_outputs = []
    padding_hidden = torch.FloatTensor([0.0 for _ in range(hidden_size)]).unsqueeze(0).to(dev)


    valid_num = num_mask.size(-1) - 2
    num_mask_only = num_mask[:, 2:]
    num_mask_only_expand =  num_mask_only.unsqueeze(-1).expand(batch_size, valid_num, hidden_size).bool()
    num_embed = num_embed.masked_fill_(~num_mask_only_expand, 0.0)

    embeddings_stacks = [[] for _ in range(batch_size)]
    loss_consistent = 0
    for t in range(max_target_length):
        num_score, op_score, current_embeddings, current_context, current_nums_embeddings, embedding_op =\
            predict(node_stacks, left_childs, text_seq_encoder, num_embed, padding_hidden, seq_mask, num_mask, op_embed = op_embed, const_embed = const_embed )
        outputs = torch.cat((op_score ,num_score), 1)
        all_node_outputs.append(outputs)

        target = target_idx[:,t].tolist()

        target_input = deepcopy(target)
        for i in range(len(target_input)):
            if target_input[i] >= len(uni_labels ):
                target_input[i] = 0

        target_input_only_op = torch.LongTensor(target_input).to(dev)

        #
        left_child, right_child, node_label = generate(current_embeddings, target_input_only_op, current_context, op_embed = embedding_op)

        left_childs = []
        for idx, l, r, node_stack, i, o, layer_rep, align, embed in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target, embeddings_stacks, batch_RE_rep, RE_tree_align, current_embeddings):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < len(uni_labels):
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

                assert len(align) == target_len[idx]
                if t < len(align):
                    index = align[t]

                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False, index))    #



            else:

                current_num = current_nums_embeddings[idx, i - len(uni_labels)].unsqueeze(0)   #
                if i > 25 : print('11111111111111', i, target)
                while len(o) > 0 and o[-1].terminal:                                           #
                    sub_stree = o.pop()
                    op = o.pop()
                    layer_align = op.align
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    RE_layer_rep = layer_rep[layer_align]
                    #
                    rep_tree = predict.project_align(current_num)
                    loss_i = 1 - torch.cosine_similarity(RE_layer_rep.float(), rep_tree.float(), dim=-1)
                    loss_consistent = loss_consistent + loss_i

                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    all_node_outputs = torch.stack(all_node_outputs, dim=1).to(dev)
    target_idx = target_idx.contiguous().to(dev)      # (batch, target-len)

    loss = masked_cross_entropy(all_node_outputs, target_idx, target_len, dev)
    loss_consistent_mean = loss_consistent / batch_size




    return loss, loss_consistent_mean




def evaluate_tree_batch(dev, predict, generate, merge, encoder_p, num_embed, const_embed, op_embed, text_seq_encoder, seq_mask, num_mask, max_length = 25, beam_size = 3):
    batch_size, hidden_size = encoder_p.size()
    uni_labels = [
        '+', '-', '-_rev', '*', '/', '/_rev'
    ]
    res_batch = []
    res_batch_score = []
    tree_batch_rep = []

    valid_num = num_mask.size(-1)-2
    num_mask_only = num_mask[:, 2:]
    num_mask_only_expand =  num_mask_only.unsqueeze(-1).expand(batch_size, valid_num, hidden_size).bool()
    num_embed = num_embed.masked_fill_(~num_mask_only_expand, 0.0)

    for batch in range(batch_size):
        seq_mask_single = seq_mask[batch].bool()
        num_mask_single = num_mask[batch].bool()
        test_res, test_score ,tree_global_rep = evaluate_tree_single_input(dev, predict, generate, merge, encoder_p[batch].unsqueeze(0), num_embed[batch].unsqueeze(0), const_embed,
                                   op_embed, text_seq_encoder[batch].unsqueeze(0), seq_mask_single.unsqueeze(0), num_mask_single.unsqueeze(0), max_length = max_length,beam_size=beam_size,num_start = len(uni_labels))

        res_batch.append(test_res)
        res_batch_score.append(test_score)
        tree_batch_rep.append(tree_global_rep)

    tree_batch_rep = torch.stack(tree_batch_rep, dim=0).squeeze()
    return res_batch, res_batch_score, tree_batch_rep






def evaluate_tree_single_input(dev, predict, generate, merge,encoder_p, num_embed,const_embed, op_embed,text_seq_encoder,seq_mask_single, num_mask_single, max_length = 45, beam_size=8, num_start=6):
    batch_size = 1
    node_stacks = [[TreeNode(_)] for _ in encoder_p.split(1, dim=0)]
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)   # (1,hidden-size)

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]


    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            left_childs = b.left_childs
            num_score, op, current_embeddings, current_context, current_nums_embeddings, embedding_op = predict(b.node_stack, left_childs, text_seq_encoder, num_embed,
                                                                                                  padding_hidden, seq_mask_single, num_mask_single,op_embed=op_embed, const_embed = const_embed )
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)
            topv, topi = out_score.topk(beam_size)

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)#
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start :
                    generate_input = torch.LongTensor([out_token])
                    generate_input = generate_input.to(dev)
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context,op_embed = embedding_op)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))
                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))

                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)  # (1,hidden-size)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))

                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)

                current_beams.append(TreeBeam(b.score + float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True

        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break
    return beams[0].out, beams[0].score, beams[0].embedding_stack[0][0].embedding








