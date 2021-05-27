_temp = __import__("testInflection.model",globals(),locals(),["A","B"])

A = _temp.A
A1 = getattr(_temp,"A")
func = getattr(A,"helloTest")
print("A = {}, A1 = {}, A.func = {}".format(A,A1,func)) #A和A1等价