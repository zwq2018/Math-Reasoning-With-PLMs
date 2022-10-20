import random

from src.utils import read_data, write_data
from typing import List, Dict
import re
from tqdm import tqdm
from collections import Counter
from copy import deepcopy
import math
def have_constant(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "1":
            return True
    return False

def have_pi(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "PI":
            return True
    return False

def have_square(target_template: List) -> bool:
    for val in target_template:
        if val.strip() == "^":
            return True
    return False


def count_variable(target_template: List) -> int:
    num_vars = set()
    for val in target_template:
        if val.strip().startswith("temp_"):
            num_vars.add(val.strip())
    return len(num_vars)

def have_multiple_m0(target_template: List):
    target_string = ' '.join(target_template)
    target_string = target_string.replace("()", "").replace("( )", "")
    target_string = re.sub(r"\(.*\)", "temp_m", target_string)
    target_template = target_string.split()
    high_priority_symbol_pos = []
    for idx, val in enumerate(target_template):
        if val in {"*", "/"}:
            high_priority_symbol_pos.append(idx)
    for prev, next in zip(high_priority_symbol_pos[:-1], high_priority_symbol_pos[1:]):
        if next - prev != 2:
            return True
    return False

def check_in_labels(current_tuple, labels):
    if current_tuple in labels:
        return current_tuple
    if current_tuple[-1] in {'+', '*'} and [current_tuple[1], current_tuple[0], current_tuple[-1]] in labels:
        return [current_tuple[1], current_tuple[0], current_tuple[-1]]
    return None

def get_labels(target_norm_post_template: List, target_template: List, remove_duplicate: bool = False):
    assert target_norm_post_template[:2] == ["x", "="]
    if len(target_norm_post_template) == 3:
        assert target_norm_post_template[2].startswith("temp_")
        target_norm_post_template.append("1")
        target_norm_post_template.append("*")
    stack = []
    pointer = 2
    labels = []
    both_m = False
    eq_2_m = {}
    got_duplicate = False
    while pointer != len(target_norm_post_template):
        stack.append(target_norm_post_template[pointer])
        if stack[-1] in {'+', '-', '*', '/', '^'}:
            if len(stack[-3:]) == 3:
                if stack[-3].startswith("m_") and stack[-2].startswith("m_"):
                    both_m = True
                if remove_duplicate:
                    checker = check_in_labels([stack[-3], stack[-2], stack[-1]], labels)
                    if checker:
                        got_duplicate = True
                        m_string = eq_2_m[' '.join(checker)]
                    else:
                        labels.append([stack[-3], stack[-2], stack[-1]])
                        m_string = f"m_{len(labels)}"
                        eq_2_m[' '.join([stack[-3], stack[-2], stack[-1]])] = m_string
                else:
                    labels.append([stack[-3], stack[-2], stack[-1]])
                    m_string = f"m_{len(labels)}"
                stack.pop()
                stack.pop()
                stack.pop()
                stack.append(m_string)
        pointer += 1
    for i, (left, right, op) in enumerate(labels):

        if left.startswith("m_") or right.startswith("m_"):
            if left.startswith("m_") and right.startswith("m_"):
                left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
            elif right.startswith("m_"):
                modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                labels[i] = [right, left, modified_op]
        else:
            if left.startswith("temp_") or right.startswith("temp_"):
                if left.startswith("temp_") and right.startswith("temp_"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("temp_"):
                    modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                    labels[i] = [right, left, modified_op]
                else:
                    assert right in {"1", "PI"}
            else:
                pass
                # raise NotImplementedError(f"all constant for label: {labels[i]}")

    for i, (left, right, op) in enumerate(labels):
        left = left[-1:] if left.startswith("temp_") else left
        right = right[-1:] if right.startswith("temp_") else right
        labels[i] = [left, right, op]

    max_temp_org = max([v for v in target_template if v.startswith("temp_")])
    max_temp_update = max([v for v in target_norm_post_template if v.startswith("temp_")])
    gap = ord(max_temp_org[-1]) - ord(max_temp_update[-1])
    if gap > 0:
        for i, (left, right, op) in enumerate(labels):
            left = chr(ord(left) + gap) if len(left) == 1 and ord(left) >= ord('a') and ord(left) <= ord('z') else left
            right = chr(ord(right) + gap) if len(right) == 1 and ord(right) >= ord('a') and ord(right) <= ord('z') else right
            labels[i] = [left, right, op]
    return labels, both_m, gap, got_duplicate




def get_labels_with_align(target_norm_post_template: List, target_template: List, remove_duplicate: bool = False, obj =None):
    assert target_norm_post_template[:2] == ["x", "="]
    if len(target_norm_post_template) == 3:
        assert target_norm_post_template[2].startswith("temp_")
        target_norm_post_template.append("1")
        target_norm_post_template.append("*")

    prefix_temp = from_infix_to_prefix(target_template[2:])
    if len(prefix_temp) == 1:                         #
        assert prefix_temp[0].startswith("temp_")
        prefix_temp=["*","1"] + prefix_temp



    assert  len(prefix_temp)==len(target_norm_post_template)-2


    stack = []
    pointer = len(prefix_temp)-1
    labels = []
    both_m = False
    eq_2_m = {}
    got_duplicate = False
    alig=[-1]*len(prefix_temp)
    while pointer >= 0:
        stack.append(prefix_temp[pointer])
        if stack[-1] in {'+', '-', '*', '/', '^'}:
            if len(stack[-3:]) == 3:
                if stack[-3].startswith("m_") and stack[-2].startswith("m_"):
                    both_m = True
                if remove_duplicate:
                    checker = check_in_labels([stack[-2], stack[-3], stack[-1]], labels)
                    if checker:
                        got_duplicate = True
                        m_string = eq_2_m[' '.join(checker)]
                        alig[pointer] = int(m_string[2:]) - 1
                    else:
                        labels.append([stack[-2], stack[-3], stack[-1]])
                        m_string = f"m_{len(labels)}"
                        eq_2_m[' '.join([stack[-2], stack[-3], stack[-1]])] = m_string
                        alig[pointer] = len(labels)-1

                else:
                    labels.append([stack[-3], stack[-2], stack[-1]])
                    m_string = f"m_{len(labels)}"
                stack.pop()
                stack.pop()
                stack.pop()
                stack.append(m_string)
        pointer -= 1
    for i, (left, right, op) in enumerate(labels):
        # left = left[-1:] if left.startswith("temp_") else left

        if left.startswith("m_") or right.startswith("m_"):
            if left.startswith("m_") and right.startswith("m_"):
                left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
            elif right.startswith("m_"):
                modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                labels[i] = [right, left, modified_op]
        else:
            if left.startswith("temp_") or right.startswith("temp_"):
                if left.startswith("temp_") and right.startswith("temp_"):
                    left_is_smaller = (ord(left[-1:]) - ord(right[-1:])) <= 0
                    modified_op = op + "_rev" if op in {'-', '/', '^'} and (not left_is_smaller) else op
                    labels[i] = [left, right, modified_op] if left_is_smaller else [right, left, modified_op]
                elif right.startswith("temp_"):
                    modified_op = op + "_rev" if op in {'-', '/', '^'} else op
                    labels[i] = [right, left, modified_op]
                else:
                    print(labels, (left, right, op), target_template, obj["id"])

                    assert right in {"1", "PI","3.14","pi"}
            else:
                pass

    for i, (left, right, op) in enumerate(labels):
        left = left[-1:] if left.startswith("temp_") else left
        right = right[-1:] if right.startswith("temp_") else right
        labels[i] = [left, right, op]

    max_temp_org = max([v for v in target_template if v.startswith("temp_")])
    max_temp_update = max([v for v in target_norm_post_template if v.startswith("temp_")])
    gap = ord(max_temp_org[-1]) - ord(max_temp_update[-1])
    if gap > 0:
        for i, (left, right, op) in enumerate(labels):
            left = chr(ord(left) + gap) if len(left) == 1 and ord(left) >= ord('a') and ord(left) <= ord('z') else left
            right = chr(ord(right) + gap) if len(right) == 1 and ord(right) >= ord('a') and ord(right) <= ord('z') else right
            labels[i] = [left, right, op]
    return labels, both_m, gap, got_duplicate, alig



def check_intermediate_m_in_order(labels: List[List[str]]):
    current_m_idx = 0
    for idx, (left_var, right_var, op) in enumerate(labels):
        if left_var.startswith("m_"):
            # try:
            assert int(left_var[2:]) - current_m_idx == 1
            # except:
            #     print("not incremental")
            current_m_idx += 1
    return True


def process_obj(obj: Dict, remove_duplicate: bool = False):     #
    target_template = [val.strip() for val in obj["target_template"]]
    labels, have_both_m, gap, got_duplicate, align = get_labels_with_align(obj["target_norm_post_template"], obj["target_template"], remove_duplicate, obj)

    type_str = "legal"


    if have_square(target_template): ##
        type_str = "have square"
        return type_str, labels, gap, False,align



    return type_str, labels, gap, got_duplicate, align

def main():
    remove_duplicate = True
    for in_file in [ "test_1.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        if remove_duplicate:
            out_file = in_file.split(".json")[0] + "multi_view_align.json"
        else:
            out_file = in_file.split(".json")[0] + "_all.json"
        data = read_data(in_file)
        count = Counter()
        inst_num_with_gap = 0
        duplicate_num = 0
        temp = []
        add_new_flag = 1
        for obj in tqdm(data, desc="processing data", total=len(data)):
            normal_equ = obj['target_template'][2:]
            res_multi_equ, change_flag = trans_equ(normal_equ)


            prefix_multi_equ = from_infix_to_prefix(res_multi_equ)
            prefix_normal_equ = from_infix_to_prefix(obj['target_template'][2:])
            if '^' not in prefix_normal_equ:
                num_list = obj['num_list']
                ans_gold = obj['answer']
                prefix_multi_equ_num = out_expression_list(prefix_multi_equ, num_list)
                prefix_normal_equ_num = out_expression_list(prefix_normal_equ, num_list)
                ans1_multi = compute_prefix_expression(prefix_multi_equ_num)
                ans2_normal = compute_prefix_expression(prefix_normal_equ_num)



            type_str, labels, gap, got_duplicate ,algin = process_obj(obj, remove_duplicate=remove_duplicate)


            if len(labels) == 0:
                assert len(obj["target_norm_post_template"]) == 3
                print("something", obj["num_list"], obj["equation"])
            if gap > 0:
                inst_num_with_gap += 1


            count[type_str] += 1
            obj["type_str"] = type_str
            obj["equation_layer"] = labels
            obj['copy'] = 'origin equation'
            obj['align'] = algin
            obj['duplicate'] = got_duplicate

            duplicate_num += 1 if got_duplicate else 0
            if change_flag and  math.fabs(ans1_multi - ans2_normal) <= 1e-4 and add_new_flag:
                count['multi_equ_infix'] += 1
                obj_new = deepcopy(obj)
                obj_new['copy'] = 'new equation'
                obj_new["target_template"][2:] = res_multi_equ
                obj_new["target_norm_post_template"][2:] = from_infix_to_postfix(res_multi_equ)

                labels_multi_equ, _, _, got_duplicate_new , align_new = get_labels_with_align(obj_new["target_norm_post_template"], obj_new["target_template"],
                                                       remove_duplicate,obj_new)
                obj_new["equation_layer"] = labels_multi_equ
                obj_new["align"] = align_new
                obj_new['duplicate'] = got_duplicate_new
                duplicate_num += 1 if got_duplicate_new else 0

                temp.append(obj)
                temp.append(obj_new)
            else:
                temp.append(obj)

        write_data(file=out_file, data = temp)

        print(inst_num_with_gap)
        print(f" duplication number: {duplicate_num}")
        for key in count:
            print(f"{key}, valid number: {count[key]}, total: {len(temp)}, %: {count[key] * 1.0 / len(data) * 100:.2f}")

def get_five_folds():
    random.seed(42)
    import os
    all_data = []
    for in_file in ["combined_train23k_processed.json", "test23k_processed.json"]:
        print(f"working on... {in_file}")
        in_file = f"../data/math23k/{in_file}"
        all_data.append(read_data(in_file))
    all_data = all_data[0] + all_data[1]
    random.shuffle(all_data)
    num_fold = 5
    fold_size = len(all_data) // num_fold
    output_folder= "math23k_five_fold"
    os.makedirs(f"../data/math23k_five_fold", exist_ok=True)
    for i in range(num_fold):
        if i == num_fold - 1:
            test_data = all_data[i * fold_size:]
            train_data = all_data[:i * fold_size]
        else:
            test_data = all_data[i * fold_size:(i + 1) * fold_size]
            train_data = all_data[:i * fold_size] + all_data[(i + 1) * fold_size:]
        size = len(train_data) + len(test_data)
        print(f"total size : {size}, train: {len(train_data)}, test: {len(test_data)}")
        write_data(file=f"../data/{output_folder}/train_{i}.json", data=train_data)
        write_data(file=f"../data/{output_folder}/test_{i}.json", data=test_data)



def trans_equ(exa):
    add_sub = ['+', '-']
    mul_div = ['*', '/']
    lp = ['(', '[']
    rp = [')', ']']
    old_exa =deepcopy(exa)

    change_flag = False
    for i, x in enumerate(exa):

        if x == '(' and exa[i + 2] in add_sub and exa[i + 4] in rp:
            left = exa[i + 1]
            op = exa[i + 2]
            right = exa[i + 3]

            if i - 2 >= 0 and exa[i - 1] in mul_div and len(exa) > i + 5 and exa[i + 5] in mul_div:  
                break
            if exa[i-3] =='/' and exa[i-1] =="*" :
                break
            elif i - 2 >= 0 and exa[i - 2] not in rp and exa[i - 1] =='*' :
                if len(exa)>i+5 and exa[i+5] not in mul_div :   #
                    temp = ['(',exa[i - 2], exa[i - 1], left, op, exa[i - 2], exa[i - 1], right,')']
                    exa = exa[:i - 2] + temp + exa[i + 5:]
                    change_flag = True
                elif len(exa)<=i+5:
                    temp = ['(',exa[i - 2], exa[i - 1], left, op, exa[i - 2], exa[i - 1], right,')']
                    exa = exa[:i - 2] + temp
                    change_flag = True

            elif len(exa)>i+5 and exa[i + 5] in mul_div and exa[i + 6] not in lp and exa[i - 1] not in mul_div:
                temp = ['(',left, exa[i + 5], exa[i + 6], op, right, exa[i + 5], exa[i + 6],')']
                exa = exa[:i] + temp + exa[i + 7:]
                change_flag = True

    #
    # if change_flag:
    #     print('exa', old_exa)
    #     print('newL',exa)
    return exa, change_flag


def from_infix_to_prefix(exp):
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
            res.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res

def from_infix_to_postfix(expression):
    st = list()
    res = list()
    priority = {"+": 0, "-": 0, "*": 1, "/": 1, "^": 2}
    for e in expression:
        if e in ["(", "["]:
            st.append(e)
        elif e == ")":
            c = st.pop()
            while c != "(":
                res.append(c)
                c = st.pop()
        elif e == "]":
            c = st.pop()
            while c != "[":
                res.append(c)
                c = st.pop()
        elif e in priority:
            while len(st) > 0 and st[-1] not in ["(", "["] and priority[e] <= priority[st[-1]]:
                res.append(st.pop())
            st.append(e)
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    return res


def out_expression_list(equ, num_list):
    res=[]
    for x in equ:
        if x.startswith('temp_'):
            xx = x[-1]
            idx = ord(str(xx))-ord(str('a'))
            num = num_list[idx]
            res.append(str(num))
        else:
            res.append(str(x))
    return res

def compute_prefix_expression(pre_fix):
    st = list()
    operators = ["+", "-", "^", "*", "/"]
    pre_fix = deepcopy(pre_fix)
    pre_fix.reverse()
    for p in pre_fix:
        if p not in operators:
            pos = re.search("\d+\(", p)
            if pos:
                st.append(eval(p[pos.start(): pos.end() - 1] + "+" + p[pos.end() - 1:]))
            elif p[-1] == "%":
                st.append(float(p[:-1]) / 100)
            elif p=='PI'or p=='pi':
                st.append(eval('3.14'))
            else:
                st.append(eval(p))
        elif p == "+" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a + b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "*" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a * b)
        elif p == "/" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if b == 0:
                return None
            st.append(a / b)
        elif p == "-" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            st.append(a - b)
        elif p == "^" and len(st) > 1:
            a = st.pop()
            b = st.pop()
            if float(eval(b)) != 2.0 or float(eval(b)) != 3.0:
                return None
            st.append(a ** b)
        else:
            return None
    if len(st) == 1:
        return st.pop()
    return None

def Merge(dict1, dict2):
    res = {**dict1, **dict2}
    return res


def create_combined_train23k_processed():
    out_file = f"../data/math23k/combined_train23k_processed.json"
    data_all=[]
    for in_file in ["train23k_processed.json"]:
        in_file = f"../data/math23k/{in_file}"
        data = read_data(in_file)
        print(len(data))
        data_all.append(data)
    res= data_all[0]+data_all[1]
    print(len(res))
    write_data(file=out_file, data=res)





if __name__ == '__main__':

    main()