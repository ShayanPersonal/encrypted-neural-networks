import numpy as np
from vhe import *

l = 100
w = 2**40
a_bound, e_bound, t_bound = 100, 1000, 100

N = 2

T_client = get_random_matrix(N, N, t_bound)
S_client = get_secret_key(T_client)

T_server = get_random_matrix(N, N, t_bound)
S_server = get_secret_key(T_server) # Not necessary

class encrypted_network():
    def __init__(self):
        self.linear_layer = np.random.normal(0.5, 0.1, (2,1))
        self.bias = np.random.normal(0, 0.1, (1,))

    def __call__(self, x, target=None):
        return self.forward(x, target)

    def forward(self, input, target=None):
        if target is None:
            # Assume input is encrypted and don't perform gradient descent
            y = self.encrypted_dot(input) + self.encrypted_bias(input)
            #y = self.encrypted_activation(y)
        else:
            # Input is unencrypted, perform gradient descent
            y = np.dot(input, self.linear_layer) + self.bias
            #y = self.activation(y)
            deriv_lin = -(target - y) * input   # if y > 0 else 0
            deriv_bias = -(target - y)          # if y > 0 else 0
            self.bias -= 0.0001 * deriv_bias
            self.linear_layer -= 0.0001 * np.expand_dims(deriv_lin, 1)
        return y

    def activation(self, x):
        # ReLU activation function
        return np.max(x, 0)

    def encrypted_activation(self, x):
        # Encrypted ReLU
        # TODO: Figure out encrypted ReLU
        print(encrypt(S_client, np.array([0])))
        return np.max(x, encrypt(S_client, np.array([0])))

    def _scale_floats_up(self, floats):
        floats = floats * 10000
        return floats.astype(np.int64)

    def _scale_floats_down(self):
        floats = floats.astype(np.float32) / 10000
        return floats

    def encrypted_dot(self, x):
        M = inner_prod_client(T_server)
        scaled_linear = self._scale_floats_up(self.linear_layer)
        encry_linear = encrypt(T_server, scaled_linear[:, 0])
        y = inner_prod(x, encry_linear, M)
        return y

    def encrypted_bias(self, x):
        scaled_bias = self._scale_floats_up(self.bias)
        encry_bias = encrypt(T_server, scaled_bias)
        return encry_bias


def get_dataset():
    x = []
    y = []

    for i in range(2):
        for j in range(2):
            x.append([i, j])
            y.append([i and j])

    return np.array(x), np.array(y)

x_data, y_data = get_dataset()
net = encrypted_network()

for epoch in range(10001):
    for x, y in zip(x_data, y_data):
        output = net(x, target=y)
        if epoch % 10000 == 0:
            print("Epoch {}: Input {}   Output {}".format(epoch, (x,y), output))


print("Testing AND prediction with encrypted 1, 1 as input.")
test_input = np.array([1, 1])
encry_input = encrypt(T_client, test_input)
print("Decrypted output: ", decrypt(S_client, net(encry_input))[0] / 10000)

print("Testing AND prediction with encrypted 0, 1 as input.")
test_input = np.array([0, 1])
encry_input = encrypt(T_client, test_input)
print("Decrypted output: ", decrypt(S_client, net(encry_input))[0] / 10000)

print("Testing AND prediction with encrypted 1, 0 as input.")
test_input = np.array([1, 0])
encry_input = encrypt(T_client, test_input)
print("Decrypted output: ", decrypt(S_client, net(encry_input))[0] / 10000)

print("Testing AND prediction with encrypted 0, 0 as input.")
test_input = np.array([0, 0])
encry_input = encrypt(T_client, test_input)
print("Decrypted output: ", decrypt(S_client, net(encry_input))[0] / 10000)