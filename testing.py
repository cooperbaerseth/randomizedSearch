

def add(a, b):
    print(a+b)

def wrap_add(say_this_first, f):
    print(say_this_first)
    f(2, 4)


wrap_add("I'm gonna add...", add)

