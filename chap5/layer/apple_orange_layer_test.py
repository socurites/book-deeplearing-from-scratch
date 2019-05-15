"""Fig 5-17 Example
"""
from chap5.layer.mul_layer import MulLayer
from chap5.layer.add_layer import AddLayer

apple  = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

'''Forward
'''
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
apple_orange_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(apple_orange_price, tax)

print(price)


'''Backward
'''
dprice = 1
dapple_orange_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, doragne_price = add_apple_orange_layer.backward(dapple_orange_price)
dorange, dorange_num = mul_orange_layer.backward(doragne_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num)
print(dorange, dorange_num)
print(dapple_price, doragne_price)
print(dapple_orange_price, dtax)
