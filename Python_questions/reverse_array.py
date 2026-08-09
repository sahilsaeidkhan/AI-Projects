arr= [3,4,5,6,7]

i = 0
j = len(arr)-1
temp = 0

while i<j:
    temp = arr[j]
    arr[j] = arr[i]
    arr[i] = temp 
    i += 1 
    j -= 1

print("Reversed", arr)




