individual = 32112322323231
a=str(individual)
b=list(a)
c=len(a)
d=len(b)
print(a)
print(b)
print(c)
print(d)
individual_str = str(individual)  # 不考虑括号

# 将字符串中的每个字符转换为整数，存入新列表
expanded_list = [int(digit) for digit in individual_str]
print(len(expanded_list))


