"""Population class for genetic algorithms."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp

from qdax.core.populations.population import Population
from qdax.custom_types import Fitness, Genotype, RNGKey


class GAPopulation(Population):
	"""Population class for genetic algorithms.

	Args:
	    genotypes: Genotypes of the individuals in the population.
	    fitnesses: Fitnesses of the individuals in the population.
	"""

	genotypes: Genotype
	fitnesses: Fitness

	@property
	def max_size(self) -> int:
		"""Gives the max size of the population."""
		first_leaf = jax.tree.leaves(self.genotypes)[0]
		return int(first_leaf.shape[0])

	@property
	def size(self) -> int:
		"""Gives the size of the population."""
		valid = self.fitnesses != -jnp.inf
		return int(jnp.sum(valid))

	@classmethod
	def init(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		population_size: int,
	) -> GAPopulation:
		"""Initializes the population.

		Start with default values and adds a first batch of genotypes
		to the population.

		Args:
		    genotypes: first batch of genotypes
		    fitnesses: corresponding fitnesses
		    population_size: size of the population we want to evolve

		Returns:
		    An initial population.
		"""
		# create default fitnesses
		default_fitnesses = -jnp.inf * jnp.ones(
			shape=(population_size, fitnesses.shape[-1])
		)

		# create default genotypes
		default_genotypes = jax.tree.map(
			lambda x: jnp.zeros(shape=(population_size,) + x.shape[1:]), genotypes
		)

		# create an initial population with those default values
		population = cls(genotypes=default_genotypes, fitnesses=default_fitnesses)

		new_population = population.add(genotypes, fitnesses)

		return new_population

	@partial(jax.jit, static_argnames=("num_samples",))
	def sample(self, random_key: RNGKey, num_samples: int) -> tuple[Genotype, RNGKey]:
		"""Sample genotypes from the population.

		Args:
		    random_key: a random key to handle stochasticity.
		    num_samples: the number of genotypes to sample.

		Returns:
		    The sample of genotypes.
		"""

		# prepare sampling probability
		mask = self.fitnesses != -jnp.inf
		p = jnp.any(mask, axis=-1) / jnp.sum(jnp.any(mask, axis=-1))

		# sample
		random_key, subkey = jax.random.split(random_key)
		samples = jax.tree.map(
			lambda x: jax.random.choice(
				subkey, x, shape=(num_samples,), p=p, replace=False
			),
			self.genotypes,
		)

		return samples, random_key

	@jax.jit
	def add(
		self, batch_of_genotypes: Genotype, batch_of_fitnesses: Fitness
	) -> GAPopulation:
		"""Adds a batch of genotypes to the population.

		Parents and offsprings are gathered and only the population_size
		bests are kept. The others are killed.

		Args:
		    batch_of_genotypes: new genotypes that we try to add.
		    batch_of_fitnesses: fitness of those new genotypes.

		Returns:
		    The updated population.
		"""

		# gather individuals and fitnesses
		candidates = jax.tree.map(
			lambda x, y: jnp.concatenate((x, y), axis=0),
			self.genotypes,
			batch_of_genotypes,
		)
		candidates_fitnesses = jnp.concatenate(
			(self.fitnesses, batch_of_fitnesses), axis=0
		)

		# sort by fitnesses
		indices = jnp.argsort(jnp.sum(candidates_fitnesses, axis=1))[::-1]

		# keep only the best ones
		survivor_indices = indices[: self.max_size]

		# keep only the best ones
		new_candidates = jax.tree.map(lambda x: x[survivor_indices], candidates)

		new_population = self.replace(
			genotypes=new_candidates, fitnesses=candidates_fitnesses[survivor_indices]
		)
