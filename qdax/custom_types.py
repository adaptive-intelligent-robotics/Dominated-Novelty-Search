"""Defines some types used in QDax"""

from typing import Generic, TypeAlias, TypeVar

import brax.envs
import jax
import jax.numpy as jnp
from chex import ArrayTree

# MDP types
Observation: TypeAlias = jnp.ndarray
Action: TypeAlias = jnp.ndarray
Reward: TypeAlias = jnp.ndarray
Done: TypeAlias = jnp.ndarray
EnvState: TypeAlias = brax.envs.State
Params: TypeAlias = ArrayTree

# Evolution types
StateDescriptor: TypeAlias = jnp.ndarray
Fitness: TypeAlias = jnp.ndarray
Genotype: TypeAlias = ArrayTree
Descriptor: TypeAlias = jnp.ndarray
Centroid: TypeAlias = jnp.ndarray
Spread: TypeAlias = jnp.ndarray
Gradient: TypeAlias = jnp.ndarray

Skill: TypeAlias = jnp.ndarray

ExtraScores: TypeAlias = dict[str, ArrayTree]

# Pareto fronts
T = TypeVar("T", bound=Fitness | Genotype | Descriptor | jnp.ndarray)


class ParetoFront(Generic[T]):
	def __init__(self) -> None:
		super().__init__()


Mask: TypeAlias = jnp.ndarray

# Others
RNGKey: TypeAlias = jax.Array
Metrics: TypeAlias = dict[str, jnp.ndarray]
