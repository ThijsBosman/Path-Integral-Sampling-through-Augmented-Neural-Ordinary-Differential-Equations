import jax
import jax.numpy as jnp
import flax.linen as nn
from diffrax import ODETerm, Dopri5, diffeqsolve
from utils import divergence
from functools import partial


class ResidualBlock(nn.Module):
    features: int

    def setup(self):
        self.dense1 = nn.Dense(self.features)
        self.dense2 = nn.Dense(self.features)

    def __call__(self, t, x):
        y = self.dense1(x)
        residual = y
        y = nn.elu(y)
        y = self.dense2(y)
        return residual + y


# Define the ANODE model
class ANODE(nn.Module):

    num_hidden: int
    n_blocks: int
    sample_dims: int
    aug_dims: int

    prior_std: float = 1.0
    T: float = 1.0
    n_steps: int = 100

    def setup(self):
        self.residual_blocks = [ResidualBlock(self.num_hidden) for _ in range(self.n_blocks)]
        self.out = nn.Dense(self.sample_dims + self.aug_dims, kernel_init=nn.initializers.normal(0.01))

    def __call__(self, t, phi):
        for block in self.residual_blocks:
            phi = block(t, phi)
        phi = self.out(phi)
        return phi

    def get_prior_samples(self, rng_key, n_samples):
        """Generates samples from the model's prior distribution that includes the augmented dimension. Also returns the log probability of the samples.

        Args:
            rng_key (jax.random.PRNGKey): The RNG key for drawing the samples.
            n_samples (int): The number of samples to generate.

        Returns:
            samples (jax.numpy.ndarray): The model's prior samples.
            logp (jax.numpy.ndarray): The log probability of the samples.
        """

        samples = jax.random.normal(rng_key, shape=(n_samples, self.sample_dims + self.aug_dims)) * self.prior_std
        logp = jax.scipy.stats.norm.logpdf(samples, scale=self.prior_std).sum(axis=1)

        return samples, logp

    def get_posterior_samples(self, params, rng_key, n_samples):
        """Generates samples from the model's posterior distribution q(x,a) using the prior samples and the flow. Also returns the log probability of the samples.

        Args:
            params (dict): The model's parameters.
            rng_key (jax.random.PRNGKey): The RNG key for drawing the samples.
            n_samples (int): The number of samples to generate.

        Returns:
            samples (jax.numpy.ndarray): The model's posterior samples.
            logp (jax.numpy.ndarray): The log probability of the samples.
        """

        prior_samples, prior_logp = self.get_prior_samples(rng_key, n_samples)
        post_samples, post_logp = self.follow_flow(
            params, prior_samples, prior_logp)

        return post_samples, post_logp

    @partial(jax.jit, static_argnames=['self', 'n_samples'])
    def sample(self, params, rng_key, n_samples):
        """Generates samples from the model's posterior distribution q(x) using the prior samples and the flow. Also returns the log probability of the samples.

        Args:
            params (dict): The model's parameters.
            rng_key (jax.random.PRNGKey): The RNG key for drawing the samples.
            n_samples (int): The number of samples to generate.

        Returns:
            x_samples (jax.numpy.ndarray): The model's posterior samples for the x dimensions.
            logp (jax.numpy.ndarray): The log probability of the samples.
        """
        samples, logp = self.get_posterior_samples(params, rng_key, n_samples)

        x_samples = samples[:, :self.sample_dims]
        a_samples = samples[:, self.sample_dims:]

        return x_samples, logp - jax.scipy.stats.norm.logpdf(a_samples, scale=self.prior_std).sum(axis=1)

    @partial(jax.jit, static_argnames=['self', 'reverse'])
    def follow_flow(self, params, samples, logp, reverse=False):
        """Follow the flow of the samples through the network.

        Args:
            params (dict): The model's parameters.
            samples (jax.numpy.ndarray): The samples to follow the flow for.
            logp (jax.numpy.ndarray): The prior log probability of the samples.

        Returns:
            samples (jax.numpy.ndarray): The samples after following the flow.
            logp (jax.numpy.ndarray): The log probability of the samples after following the flow.
        """

        t0 = 0.0 if not reverse else self.T
        t1 = self.T if not reverse else 0.0

        # Create the ODE term and solver
        ode_term = ODETerm(self._vector_field)
        solver = Dopri5()

        # Solve the ODE
        solution = diffeqsolve(
            ode_term,
            solver,
            t0=t0,
            t1=t1,
            dt0=(t1 - t0) / self.n_steps,
            y0=(samples, logp),
            args=params,
        )

        return solution.ys[0][0], solution.ys[1][0]

    def _vector_field(self, t, y, params):
        """Defines the vector field for the forward flow."""

        phi, _ = y
        dphi_dt = self.apply(params, t, phi)

        # Make sure to take the divergence with respect to phi
        def g(x): return self.apply(params, t, x)
        div = jax.vmap(divergence(g))(phi)

        return (dphi_dt, -div)

    @partial(jax.jit, static_argnames=['self'])
    def get_logp(self, params, samples, rng_key):
        """Calculates the log probability of samples under the model's posterior distribution q(x).

        Args:
            params (dict): The model's parameters.
            samples (jax.numpy.ndarray): The samples to calculate the log probability for.
            rng_key (jax.random.PRNGKey): The RNG key for drawing the samples.

        Returns:
            samples (jax.numpy.ndarray): The model's posterior samples.
            logp (jax.numpy.ndarray): The log probability of the samples.
        """

        # Append the augmented dimensions to the samples
        a_samples = jax.random.normal(rng_key, shape=(samples.shape[0], self.aug_dims))
        samples = jnp.concatenate([samples, a_samples], axis=-1)

        # Calculate the log probability of the samples
        samples, logp = self.follow_flow(params, samples, jnp.zeros(samples.shape[0]), reverse=True)
        return samples[:, :self.sample_dims],  jax.scipy.stats.norm.logpdf(samples, scale=self.prior_std).sum(axis=1) - jax.scipy.stats.norm.logpdf(a_samples, scale=self.prior_std).sum(axis=1) - logp


