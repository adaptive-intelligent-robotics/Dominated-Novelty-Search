from __future__ import annotations

from functools import partial

import flax.struct
import jax
import jax.numpy as jnp

from qdax.custom_types import Descriptor, Fitness, Genotype, Observation, RNGKey


class AdaptivePopulation(flax.struct.PyTreeNode):
	"""
	Class for the adaptive population.

	Args:
		genotypes: Genotypes of the individuals in the population.
		fitnesses: Fitnesses of the individuals in the population.
		descriptors: Descriptors of the individuals in the population.
		observations: Observations that the genotype gathered in the environment.
	"""

	genotypes: Genotype
	fitnesses: Fitness
	descriptors: Descriptor
	observations: Observation
	k: int = flax.struct.field(pytree_node=False)

	@property
	def max_size(self) -> int:
		"""Gives the max size of the population."""
		first_leaf = jax.tree.leaves(self.genotypes)[0]
		return first_leaf.shape[0]

	@property
	def size(self) -> int:
		"""Gives the size of the population."""
		valid = self.fitnesses != -jnp.inf
		return jnp.sum(valid)

	@classmethod
	def init(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		observations: Observation,
		max_size: int,
		k: int,
	) -> AdaptivePopulation:
		"""Initialize a population with an initial population of genotypes.

		Args:
			genotypes: initial genotypes, pytree in which leaves
				have shape (batch_size, num_features)
			fitnesses: fitness of the initial genotypes of shape (batch_size,)
			descriptors: descriptors of the initial genotypes
				of shape (batch_size, num_descriptors)
			observations: observations experienced in the evaluation task.
			size: size of the population

		Returns:
			an initialized adaptive population.
		"""

		# Init population with dummy values
		dummy_genotypes = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
			genotypes,
		)
		dummy_fitnesses = jnp.full((max_size,), fill_value=-jnp.inf)
		dummy_descriptors = jnp.full(
			(max_size,) + descriptors.shape[1:], fill_value=jnp.nan
		)
		dummy_observations = jax.tree.map(
			lambda x: jnp.full((max_size,) + x.shape[1:], fill_value=jnp.nan),
			observations,
		)

		population = AdaptivePopulation(
			genotypes=dummy_genotypes,
			fitnesses=dummy_fitnesses,
			descriptors=dummy_descriptors,
			observations=dummy_observations,
			k=k,
		)

		population = population.add(
			genotypes,
			descriptors,
			fitnesses,
			observations,
		)
		return population

	@partial(jax.jit, static_argnames=("num_samples",))
	def sample(self, random_key: RNGKey, num_samples: int) -> tuple[Genotype, RNGKey]:
		"""Sample elements in the population.

		Args:
			random_key: a jax PRNG random key
			num_samples: the number of elements to be sampled

		Returns:
			samples: a batch of genotypes sampled in the population
			random_key: an updated jax PRNG random key
		"""

		random_key, sub_key = jax.random.split(random_key)
		grid_empty = self.fitnesses == -jnp.inf
		p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

		samples = jax.tree.map(
			lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
			self.genotypes,
		)

		return samples, random_key

	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_observations: Observation,
	) -> AdaptivePopulation:
		"""Adds a batch of genotypes to the population.

		Args:
			batch_of_genotypes: genotypes of the individuals to be considered
				for addition in the population.
			batch_of_descriptors: associated descriptors.
			batch_of_fitnesses: associated fitness.
			batch_of_observations: associated observations.

		Returns:
			A new unstructured population where the relevant individuals have been
			added.
		"""
		# Concatenate everything
		genotypes = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotypes,
			batch_of_genotypes,
		)
		descriptors = jnp.concatenate([self.descriptors, batch_of_descriptors], axis=0)
		fitnesses = jnp.concatenate([self.fitnesses, batch_of_fitnesses], axis=0)
		observations = jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.observations,
			batch_of_observations,
		)

		is_empty = fitnesses == -jnp.inf

		# Fitter
		fitter = fitnesses[:, None] <= fitnesses[None, :]
		fitter = jnp.where(
			is_empty[None, :], False, fitter
		)  # empty individuals can not be fitter
		fitter = jnp.fill_diagonal(
			fitter, False, inplace=False
		)  # an individual can not be fitter than itself

		# Distance to k-fitter-nearest neighbors
		distance = jnp.linalg.norm(
			descriptors[:, None, :] - descriptors[None, :, :], axis=-1
		)
		distance = jnp.where(fitter, distance, jnp.inf)
		values, indices = jax.vmap(partial(jax.lax.top_k, k=self.k))(-distance)
		distance = jnp.mean(
			-values, where=jnp.take_along_axis(fitter, indices, axis=1), axis=-1
		)  # if number of fitter individuals is less than k, top_k will return at least one inf
		distance = jnp.where(
			jnp.isnan(distance), jnp.inf, distance
		)  # if no individual is fitter, set distance to inf
		distance = jnp.where(
			is_empty, -jnp.inf, distance
		)  # empty cells have distance -inf

		# Sort by distance to k-fitter-nearest neighbors
		indices = jnp.argsort(distance, descending=True)
		indices = indices[: self.max_size]
		is_offspring_added = jax.vmap(lambda i: jnp.any(indices == i))(
			jnp.arange(self.max_size, self.max_size + batch_of_fitnesses.size)
		)

		# Sort
		genotypes = jax.tree.map(lambda x: x[indices], genotypes)
		descriptors = descriptors[indices]
		fitnesses = fitnesses[indices]
		observations = jax.tree.map(lambda x: x[indices], observations)

		return AdaptivePopulation(
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			observations=observations,
			k=self.k,
		)  # , is_offspring_added
