from typing import Optional

import jax
import jax.numpy as jnp

from kheperax.envs.scoring import create_kheperax_scoring_fn, get_final_state_desc
from kheperax.envs.wrappers import EpisodeWrapper
from kheperax.tasks.config import KheperaxConfig
from kheperax.tasks.main import KheperaxTask
from mypy.plugins.default import partial

from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.custom_types import RNGKey, Descriptor


def get_n_spaced_state_desc(data: QDTransition, mask: jax.typing.ArrayLike, n_points: int) -> Descriptor:
    """Compute N equally spaced state descriptors along the trajectory.

    Args:
        data: QDTransition containing state descriptors
        mask: Binary mask indicating valid timesteps (1 for padded/invalid, 0 for valid)
        n_points: Number of equally spaced points to sample (including start and end)

    Returns:
        Array of N state descriptors sampled at equal intervals along the trajectory
    """
    # Get trajectory lengths for each batch element
    traj_lengths = jnp.sum(1.0 - mask, axis=1)

    def get_spaced_indices(length):
        # Get equally spaced indices for a single trajectory
        # We subtract 1 from length since indices are 0-based
        indices = jnp.linspace(0, length - 1, n_points)
        return jnp.floor(indices).astype(jnp.int32)

    # Vectorize over batch dimension
    indices = jax.vmap(get_spaced_indices)(traj_lengths)

    # Get state descriptors at computed indices
    descriptors = jax.vmap(lambda x, idx: x[idx])(data.state_desc, indices)

    return jnp.reshape(descriptors, (descriptors.shape[0], -1))


def create_kheperax_energy_task(
    kheperax_config: KheperaxConfig,
    num_points_desc: Optional[int] = None,
):
    env = KheperaxTask(kheperax_config)
    ep_wrapper_env = EpisodeWrapper(
        env,
        kheperax_config.episode_length,
        action_repeat=kheperax_config.action_repeat,
    )

    # Init policy network
    policy_layer_sizes = kheperax_config.mlp_policy_hidden_layer_sizes + (
        ep_wrapper_env.action_size,
    )
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    if num_points_desc is None:
        bd_extraction_fn = get_final_state_desc
    else:
        bd_extraction_fn = partial(get_n_spaced_state_desc, n_points=num_points_desc)

    scoring_fn = create_kheperax_scoring_fn(
        ep_wrapper_env,
        policy_network,
        bd_extraction_fn,
        episode_length=kheperax_config.episode_length,
    )

    return ep_wrapper_env, policy_network, scoring_fn
