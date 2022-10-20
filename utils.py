

import re
from src.utils import read_data, write_data
from tqdm import tqdm

def count_num_operations(file:str):

    num_operations = 0
    total_insts = 0
    pattern = r"[+-/\*]"
    with open(file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            if line.startswith("id:") and "equation:" in line:
                equation = line.split(",")[1].split(":")[1].strip()
                total_insts += 1
                operations = re.findall(pattern, equation)
                operations
                num_operations += len(operations)
                if len(operations) > 2:
                    print(equation)


    print(f" total number of instances: {total_insts}, average operation: {num_operations * 1.0/total_insts}")

def check_json_data(file:str):
    data =read_data(file=file)
    num_3_var = 0
    filtered_data = []
    for obj in tqdm(data, total = len(data)):
        if len(obj['variables']) == 4 or len(obj['variables']) == 3:
            num_3_var+= 1
            filtered_data.append(obj)
    print(num_3_var, len(data), num_3_var*1.0/len(data)*100)
    print("soemthing")
    num_train = int(len(filtered_data) * 0.9)
    import random
    random.seed(42)
    # random.shuffle(filtered_data)
    write_data(file="data/simple_cases_train_all.json", data=filtered_data[:num_train])
    write_data(file="data/simple_cases_test_all.json", data=filtered_data[num_train:])
    # write_data(file="data/more_variables.json", data=filtered_data)
    print(f"num train:{num_train}, testing: {len(filtered_data[num_train:])}")


def split_generation():
    src_data_file = "data/src_data.json"
    tgt_data_file = "data/tgt_data.json"
    src_data = read_data(file=src_data_file)
    tgt_data = read_data(file=tgt_data_file)
    insts = []
    for src,tgt in zip(src_data, tgt_data):
        src_text = ''
        for token in src:
            if token in ['+', '-', '*', '/']:
                src_text += f' {token} '
            elif token.startswith("<") and token.endswith(">"):
                src_text += f' {token} '
            else:
                src_text += token
        tgt_text = ''
        for token in tgt:
            if token.startswith("<") and token.endswith(">"):
                tgt_text += f' {token} '
            else:
                tgt_text += token
        insts.append({
            "src": src_text,
            "tgt": tgt_text
        })
    import random
    random.seed(42)
    random.shuffle(insts)
    train_num = int(len(insts)*0.9)

    trains = insts[:train_num]
    validations = insts[train_num:]
    write_data(file="data/gen_train.json", data=trains)
    write_data(file="data/gen_val.json", data=validations)

def check_4_variables():
    # file = "data/four_var_cases_updated.json"
    # data = read_data(file=file)
    # write_data(file=file, data=data)

    ## split four variables:
    import random
    random.seed(42)
    file = "data/all_generated_1.0_updated.json"
    data =read_data(file= file)
    random.shuffle(data)
    train_num = int(len(data) * 0.9)
    train_data = data[:train_num]
    test_data = data[train_num:]
    write_data(file="data/fv_train_updated.json", data=train_data)
    write_data(file="data/fv_test_updated.json", data=test_data)


def split_complext():
    import random
    random.seed(42)
    file = "data/complex/mwp_processed.json"
    data = read_data(file=file)
    random.shuffle(data)
    train_num = int(len(data) * 0.9)
    train_data = data[:train_num]
    test_data = data[train_num:]
    write_data(file="data/complex/train.json", data=train_data)
    write_data(file="data/complex/validation.json", data=test_data)


if __name__ == '__main__':
    # count_num_operations("data/cate_res_comp.txt")
    # check_json_data(file="data/simple_cases.json")
    # split_generation()
    # check_4_variables()
    split_complext()