import jax 
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import equinox as eqx
from typing import List

N_SAMPLES = 200
N_EPOCHS = 30_000
LAYERS = [1, 10, 10, 10, 1]
LR = .1

key = jax.random.PRNGKey(0)
key, xkey, ynoisekey = jax.random.split(key, 3)
x_samples = jax.random.uniform(xkey, (N_SAMPLES, 1), minval = 0.0, maxval = 2*jnp.pi)
y_samples = jnp.sin(x_samples) + jax.random.normal(ynoisekey, (N_SAMPLES, 1))*0.3

class SimpleMLP(eqx.Module):
    layers : List[eqx.nn.Linear]
    def __init__(self, layer_size, key):
        self.layers = []

        for (fan_in, fan_out) in zip(layer_size[:-1], layer_size[1:]):
            key, subkey = jax.random.split(key)
            self.layers.append(
                eqx.nn.Linear(fan_in, fan_out, use_bias=True, key=subkey)

            )

    def __call__(self, x):
        a = x 
        for layer in self.layers[:-1]:
            a = jax.nn.sigmoid(layer(a)) # no sigmoid at the last layer
        a = self.layers[-1](a)
        return a

model = SimpleMLP(LAYERS, key = key)

# init pred
plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, jax.vmap(model)(x_samples) )
plt.show()

# loss was created here as least_square
def model_to_loss(m, x, y):
    prediction = jax.vmap(m)(x)
    delta = prediction - y
    loss = jnp.mean(delta**2)
    return loss

# define a loss function
model_to_loss(model, x_samples, y_samples)


#model_to_loss_and_grad = jax.value_and_grad(model_to_loss)
# equinox version of grad_fn 
model_to_loss_and_grad = eqx.filter_value_and_grad(model_to_loss)

# apply optimizers
opt = optax.sgd(LR)
opt_state = opt.init(eqx.filter(model, eqx.is_array)) # only opt on array entries

@eqx.filter_jit
def make_step(m, opt_state, x, y):
    loss, grad = model_to_loss_and_grad(m, x, y)
    updates, opt_state = opt.update(grad, opt_state, m) # update the optimizer status
    m = eqx.apply_updates(m, updates)
    return m, opt_state, loss

loss_history = []
for epoch in range(N_EPOCHS):
    model, opt_state, loss = make_step(model, opt_state, x_samples, y_samples)
    loss_history.append(loss)

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss {loss}")

plt.plot(loss_history)
plt.yscale("log")
plt.show()

plt.scatter(x_samples, y_samples)
plt.scatter(x_samples, jax.vmap(model)(x_samples))
plt.title("eval")
plt.show()
