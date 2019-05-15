from chap5.layer.mul_layer import MulLayer

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

"""Forward
"""
apple_pice = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_pice, tax)

print(price)


"""Backward
"""
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_num, dtax)