def add(a: int, b: int) -> int:
    return a + b


# for문으로 1-100까지 숫자를 add함수로 더하기
result = 0
for i in range(1, 101):
    result = add(result, i)
print(result)
