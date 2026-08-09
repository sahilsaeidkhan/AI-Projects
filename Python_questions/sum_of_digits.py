n = 4567
total = 0

while n > 0:
    total = n % 10 + total
    n = n // 10

print(f"The sum of 4567 is {total}")