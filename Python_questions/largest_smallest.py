arr = [ 3 ,4 , 5 ,3 , 7]
# find largest and smallest element 


maximum = arr[0]
minimum = arr[0]


for x in arr:
    if x > maximum:
        maximum = x

    if x < minimum:
        minimum = x

print(f"Maximum is {maximum} and minimum is {minimum}")