import random

def count_pairs(nums):
    count = 0
    n =len(nums)

    for i in range(n):
        for j in range(i+1 ,n):
            if nums[i]+nums[j] ==10:
                count=count +1
    return count

def find_range(values):
    if len(values) < 3:
        return "Range determination not possible "
    minimum_val =values[0]
    maximum_val =values[0]

    for v in values :
        if v < minimum_val:
            minimum_val =v
        if v> maximum_val:
            maximum_val =v
    return maximum_val - minimum_val

def multiply(A,B):
    n=len(A)
    result =[[0] *n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            for k in range(n):
                result[i][j] =result[i][j] +A[i][k] * B[k][j]
    return result

def power(A,m):
    result =A

    for i in range(1,m):
        result = multiply(result ,A)
    return result

def highest_char(text):
    freq={}
    for ch in text:
        if ch.isalpha():
            ch=ch.lower()
            if ch in freq:
                freq[ch] =freq[ch]+1
            else:
                freq[ch] =1
    max_char=""
    max_count=0
    for ch in freq:
        if freq[ch]> max_count:
            max_char =ch
            max_count =freq[ch]
    return max_char,max_count

def mean_median_mode(nums):
    total =0
    for n in nums:
        total =total +n
        mean = total /len(nums)
        nums.sort()
        median =nums[len(nums)//2]

        freq ={}
        for n in nums:
            if n in freq:
                freq[n]=freq[n]+1
            else:
                    freq[n] = 1
        mode =nums[0]
        max_freq =freq[mode]

        for key in freq:
            if freq[key]> max_freq:
                max_freq =freq[key]
                mode =key
        return mean,median,mode


list1 =[2,7,4,1,3,6]
print("pairs:" , count_pairs(list1))
list2 =[5,3,8,1,0,4]
print("range:",find_range(list2))

A=[[1,2],[3,4]]
m=2
result =power(A,m)
print("matrix power:")
for row in result:
    print(row)

text ="hippopotamus"
ch,cnt =highest_char(text)
print("highest character:" ,ch,cnt)

numbers =[]
for i in range (25):
    numbers.append(random.randint(1,10))

mean,median,mode =mean_median_mode(numbers)
print("mmm:" , numbers , mean , median , mode )

    
     
    
