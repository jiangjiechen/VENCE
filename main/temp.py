entity_sample_pos=[0, 1, 2, 3, 4, 5]
spilt_token=[[0, 1, 2], [3, 4, 5], [9, 10]]

for _, ww in enumerate(spilt_token):
    print(_,ww)
    if set(ww).issubset(set(entity_sample_pos)):
        spilt_token.pop(_)

def remove_sublists(a, b):
    result = []
    for lst in a:
        if not all(item in b for item in lst):
            result.append(lst)
    return result


print(spilt_token)
print(remove_sublists(spilt_token,entity_sample_pos))
print(spilt_token)