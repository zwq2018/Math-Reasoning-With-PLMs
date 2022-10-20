from copy import deepcopy

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
            res.append(e[-1])
        else:
            res.append(e)
    while len(st) > 0:
        res.append(st.pop())
    res.reverse()
    return res


def trans_equ(exa):
    add_sub = ['+', '-']
    mul_div = ['*', '/']
    lp = ['(', '[']
    rp = [')', ']']
    print('exa',exa)
    for i, x in enumerate(exa):

        if x == '(' and exa[i + 2] in add_sub and exa[i + 4] in rp:
            left = exa[i + 1]
            op = exa[i + 2]
            right = exa[i + 3]
            if i-2>=0 and exa[i-1] in mul_div and len(exa)>i+5 and  exa[i+5] in mul_div:
                break

            elif i - 2 >= 0 and exa[i - 2] not in rp and exa[i - 1] in mul_div :
                if len(exa)>i+5 and exa[i+5] not in mul_div :
                    temp = [exa[i - 2], exa[i - 1], left, op, exa[i - 2], exa[i - 1], right]
                    exa = exa[:i - 2] + temp + exa[i + 5:]
                elif len(exa)<=i+5:
                    temp = [exa[i - 2], exa[i - 1], left, op, exa[i - 2], exa[i - 1], right]
                    exa = exa[:i - 2] + temp


            elif exa[i + 5] in mul_div and exa[i + 6] not in lp and exa[i - 1] not in mul_div:
                temp = [left, exa[i + 5], exa[i + 6], op, right, exa[i + 5], exa[i + 6]]
                exa = exa[:i] + temp + exa[i + 7:]
    return exa



inp=['(', 'temp_a', '/', '1', '+', 'temp_a', '/', 'temp_b', ')']

res=from_infix_to_prefix(inp)
print(res)