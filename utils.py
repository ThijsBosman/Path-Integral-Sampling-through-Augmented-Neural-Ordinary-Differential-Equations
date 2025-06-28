import jax
import jax.numpy as jnp
from functools import partial


def divergence(f: callable) -> callable:
    """Computes the divergence function.

    Args:
        f (callable): Function for which to get the divergence function for.

    Returns:
        callable: The divergence function of f.
    """

    def div(x):
        jacobian = jax.jacobian(f)(x)

        if jacobian.ndim == 0:
            return jacobian

        return jnp.trace(jacobian, axis1=-1, axis2=-2)

    return div


@partial(jax.jit, static_argnames=['target'])
def metropolis_step(samples: jnp.array,
                    p_samples: jnp.array,
                    target: callable,
                    key: jax.random.PRNGKey) -> tuple:
    """Performs the Metropolis-Hastings algorithm to sample from a target probability density function. 
    The proposal is assumed to be independent of the previous sample.

    Args:
        samples (jnp.array): The samples from the proposal distribution.
        p_samples (jnp.array): The propability of the samples from the proposal distribution.
        target (callable): The target probability density function.
        key (jax.random.PRNGKey): The random key used for the Metropolis-Hastings algorithm.

    Returns:
        tuple (jnp.array, float): The accepted samples and the acceptance rate.
    """

    def scan_fn(carry, i):
        accepted_samples, n_accepted, target_prev, proposal_prev, key = carry
        key, subkey = jax.random.split(key)

        target_current = target(samples[i])
        proposal_current = p_samples[i]

        # Determine whether to accept the new sample
        accept_ratio = jnp.minimum(
            1.0, (target_current * proposal_prev) / (target_prev * proposal_current))
        accepted = jax.random.uniform(subkey) < accept_ratio

        # Set the updated sample and proposal
        accepted_samples = accepted_samples.at[i].set(
            jnp.where(accepted, samples[i], accepted_samples[i - 1]))
        target_prev = jnp.where(accepted, target_current, target_prev)
        proposal_prev = jnp.where(accepted, proposal_current, proposal_prev)

        # Update number of accepted samples
        n_accepted = n_accepted + jnp.where(accepted, 1, 0)

        return (accepted_samples, n_accepted, target_prev, proposal_prev, key), None

    n = len(samples)

    accepted_samples = jnp.zeros_like(samples)
    accepted_samples = accepted_samples.at[0].set(samples[0])

    target_prev = target(samples[0])
    proposal_prev = p_samples[0]

    # Run the algorithm
    (accepted_samples, n_accepted, _, _, _), _ = jax.lax.scan(scan_fn,
                                                              (accepted_samples,
                                                               0,
                                                               target_prev,
                                                               proposal_prev,
                                                               key),
                                                              jnp.arange(1, n))

    return accepted_samples, n_accepted / (n - 1)


@partial(jax.jit, static_argnames=['n_samples', 'log_target'])
def calculate_loss_reverse(state, params, sample_key, n_samples, log_target) -> float:
    """Calculates the loss (reverse KL divergence) between the target and proposal distribution.

    Args:
        state (TrainState): The model state.
        params (dict): The model's parameters.
        sample_key (jax.random.PRNGKey): The RNG key for drawing the samples
        n_samples (int): The number of samples used to estimate the loss.
        log_target (callable): The log target probability density function.

    Returns:
        loss (float): The loss. 
    """

    # Get the samples from our proposed distribution
    samples, logps = state.apply_fn(params, sample_key, n_samples)

    # Calculate the logp of the samples according to the target distribution
    target_logp = jax.vmap(log_target)(samples)

    # Calculate the reverse KL divergence
    loss = jnp.mean(logps - target_logp)

    return loss


@partial(jax.jit, static_argnames=['n_samples', 'n_repetitions', 'log_target'])
def calculate_metrics_reverse(state, sample_key, n_samples, n_repetitions, log_target, loss):
    """Calculates the metrics for the model.

    Args:
        state (_type_): The model state.
        sample_key (jax.random.PRNGKey): The RNG key for drawing the samples
        n_samples (int): The number of samples used to estimate the metrics.
        log_target (callable): The log target probability density function.
        loss (float): The loss.

    Returns:
        dict: A dictionary containing the metrics.
    """

    ess = jnp.zeros((n_repetitions,))
    accept_rate = jnp.zeros((n_repetitions,))
    int_error = jnp.zeros((n_repetitions,))
    expected_loss = jnp.zeros((n_repetitions,))

    for i in range(n_repetitions):

        # Generate samples
        sample_key, metric_key = jax.random.split(sample_key)
        samples, logps = state.apply_fn(state.params, metric_key, n_samples)
        target_logp = jax.vmap(log_target)(samples)

        # ESS metric
        ratio = jnp.exp(target_logp - logps)
        ESS_estimation = (jnp.mean(ratio)**2) / jnp.mean((ratio)**2)
        ess = ess.at[i].set(ESS_estimation)

        # Acceptance rate metric
        _, ACCEPT_RATE_estimation = metropolis_step(
            samples, jnp.exp(logps),
            lambda x: jnp.exp(log_target(x)).squeeze(),
            metric_key)
        accept_rate = accept_rate.at[i].set(ACCEPT_RATE_estimation)

        # Integration error metric
        target_loss = jnp.mean(logps - target_logp)
        expected_loss = expected_loss.at[i].set(target_loss)
        int_error = int_error.at[i].set((target_loss - loss)**2)

    return {
        'ESS': jnp.mean(ess),
        'ESS_std': jnp.std(ess),
        'ACCEPT_RATE': jnp.mean(accept_rate),
        'ACCEPT_RATE_std': jnp.std(accept_rate),
        'MSE_INTEGRATE_ERROR': jnp.mean(int_error),
        'MSE_INTEGRATE_ERROR_std': jnp.std(int_error),
        'EXPECTED_LOSS': jnp.mean(expected_loss),
        'EXPECTED_LOSS_std': jnp.std(expected_loss)
    }


@partial(jax.jit, static_argnames=['n_samples', 'log_target'])
def train_step_reverse(state, train_step_key, n_samples, log_target):
    """Performs a single training step using the reverse KL divergence.

    Args:
        state (TrainState): The model state.
        epoch (int): The current epoch number.
        train_step_key (_type_): The RNG used for the training step
        n_samples (_type_): The number of samples used to estimate the loss
        log_target (_type_): The log target probability density function.

    Returns:
        tuple (state, dict): The model state after the training step and the train metrics.
    """

    # Get the gradient function
    grad_fn = jax.value_and_grad(calculate_loss_reverse, argnums=1)

    # Determine gradients
    train_step_key, grad_key = jax.random.split(train_step_key)
    loss, grads = grad_fn(state, state.params, grad_key, n_samples, log_target)

    # Update the parameters
    state = state.apply_gradients(grads=grads)

    return state, loss
        

def train_reverse(state, train_key, log_target, num_epochs=1000, batch_size=256):
    """Train function for the NODE model.

    Args:
        state (_type_): The model state we wish to train.
        train_key (_type_): The RNG key used for the training process.
        log_target (callable): The log target probability density function.
        num_epochs (int): Number of epochs to train the model for. Defaults to 1000.
        batch_size (int, optional): The batch size. Defaults to 256.
        verbose (bool, optional): Whether to print the training progress. Defaults to True.
        calculate_metrics (bool, optional): Whether to calculate the metrics during training. Defaults to True.

    Returns:
        _type_: The trained model state and the training metrics.
    """

    train_metrics = {'epoch': [],
        'loss': [],
        'ESS': [],
        'ESS_std': [],
        'ACCEPT_RATE': [],
        'ACCEPT_RATE_std': [],
        'MSE_INTEGRATE_ERROR': [],
        'MSE_INTEGRATE_ERROR_std': [],
        'EXPECTED_LOSS': [],
        'EXPECTED_LOSS_std': []
    }

    for epoch in range(num_epochs):

        # Generate key for the training step
        train_key, train_step_key = jax.random.split(train_key)

        # Apply a training step
        state, loss = train_step_reverse(
            state, train_step_key, batch_size, log_target)
        log_message = f"Epoch {epoch}: Loss: {loss:.4f}"

        # Update the training metrics
        train_metrics['loss'].append(loss)

        if epoch % 250 == 0:
            train_key, metric_key = jax.random.split(train_key)

            current_metrics = calculate_metrics_reverse(
                state, metric_key, 8 * batch_size, 20, log_target, loss)

            # Update the training metrics
            train_metrics['epoch'].append(epoch)
            train_metrics['ESS'].append(current_metrics['ESS'])
            train_metrics['ESS_std'].append(current_metrics['ESS_std'])
            train_metrics['ACCEPT_RATE'].append(current_metrics['ACCEPT_RATE'])
            train_metrics['ACCEPT_RATE_std'].append(current_metrics['ACCEPT_RATE_std'])
            train_metrics['MSE_INTEGRATE_ERROR'].append(current_metrics['MSE_INTEGRATE_ERROR'])
            train_metrics['MSE_INTEGRATE_ERROR_std'].append(current_metrics['MSE_INTEGRATE_ERROR_std'])
            train_metrics['EXPECTED_LOSS'].append(current_metrics['EXPECTED_LOSS'])
            train_metrics['EXPECTED_LOSS_std'].append(current_metrics['EXPECTED_LOSS_std'])

            # Log the metrics
            log_message += f", ESS: {current_metrics['ESS']:.4f} ± {current_metrics['ESS_std']:.4f}"
            log_message += f", ACCEPT_RATE: {current_metrics['ACCEPT_RATE']:.4f} ± {current_metrics['ACCEPT_RATE_std']:.4f}"
            log_message += f", INTEGRATE_ERROR: {current_metrics['MSE_INTEGRATE_ERROR']:.4f} ± {current_metrics['MSE_INTEGRATE_ERROR_std']:.4f}"

        # Log the training progress
        print(log_message)

    # Convert everything to jax arrays
    for key in train_metrics:
        train_metrics[key] = jnp.array(train_metrics[key])

    return state, train_metrics
