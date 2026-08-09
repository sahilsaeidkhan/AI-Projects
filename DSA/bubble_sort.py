arr = [ 6,4,5,1,2,0]


for i in range(len(arr)):
    for j in range(len(arr)-1):
        if arr[j] > arr[j+1]:
            # swap
            temp = arr[j]
            arr[j] = arr[j+1]
            arr[j+1] = temp

print("Sorted Array" , arr)
