from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

digits = load_digits()
x_train, x_test, t_train, t_test = train_test_split(digits.data, digits.target, random_state=42)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
