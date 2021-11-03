import os


def clear(): return os.system("cls")


clear()


# def remainder(num):
# return num % 2


# print(remainder(5))

#def remainder(num): return num % 2
# print(type(remainder))
# print(remainder(5))


#def product(x, y): return x * y

#def product(x, y): return x * y

#print(product(2, 3))


def myfunction(num):
    return lambda x: x * num


result10 = myfunction(10)
result100 = myfunction(100)
#result10 = lambda x: x * 10

# print(result10(9))


def myfunc(n):
    return lambda a: a * n


mydoubler = myfunc(2)  # this is a function with n
mytripler = myfunc(3)

# print(mydoubler(11))  # this is a


# filter
numbers = [2, 4, 6, 8, 10, 3, 18, 14, 21]

filterd_list = list(filter(lambda num: (num > 7), numbers))

# print(filterd_list)


# Map

Mapped_list = list(map(lambda num: num % 2, numbers))

# print(Mapped_list)


def x(a): return a + 10

print(x(5))
