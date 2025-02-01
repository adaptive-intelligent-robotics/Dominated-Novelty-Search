"""Core components of the MAP-Elites algorithm."""

from __future__ import annotations

from collections.abc import Callable
from functools import partial
from typing import Any

import jax

from qdax.core.emitters.emitter import Emitter, EmitterState
from qdax.core.populations.population import Population
from qdax.custom_types import (
	Descriptor,
	ExtraScores,
	Fitness,
	Genotype,
	Metrics,
	RNGKey,
)


class MAPElites:
	"""Core elements of the MAP-Elites algorithm.

	Note: Although very similar to the GeneticAlgorithm, we decided to keep the
	MAPElites class independant of the GeneticAlgorithm class at the moment to keep
	elements explicit.

	Args:
		scoring_function: a function that takes a batch of genotypes and compute
			their fitnesses and descriptors
		emitter: an emitter is used to suggest offsprings given a MAPELites
			population. It has two compulsory functions. A function that takes
			emits a new population, and a function that update the internal state
			of the emitter.
		metrics_function: a function that takes a MAP-Elites population and compute
			any useful metric to track its evolution
	"""

	def __init__(
		self,
		scoring_function: Callable[
			[Genotype, RNGKey], tuple[Fitness, Descriptor, ExtraScores, RNGKey]
		],
		emitter: Emitter,
		metrics_function: Callable[[Population], Metrics],
	) -> None:
		self._scoring_function = scoring_function
		self._emitter = emitter
		self._metrics_function = metrics_function

	@partial(jax.jit, static_argnames=("self",))
	def init(
		self,
		population: Population,
		genotypes: Genotype,
		random_key: RNGKey,
	) -> tuple[Population, EmitterState | None, RNGKey]:
		"""
		Initialize a Map-Elites population with an initial population of genotypes.
		Requires the definition of centroids that can be computed with any method
		such as CVT or Euclidean mapping.

		Args:
			genotypes: initial genotypes, pytree in which leaves
				have shape (batch_size, num_features)
			population: the MAP-Elites population initialized alreadys
			random_key: a random key used for stochastic operations.

		Returns:
			An initialized MAP-Elite population with the initial state of the emitter,
			and a random key.
		"""
		# score initial genotypes
		fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
			genotypes, random_key
		)

		# get initial state of the emitter
		emitter_state, random_key = self._emitter.init(
			random_key=random_key,
			repertoire=population,
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores=extra_scores,
		)

		return population, emitter_state, random_key

	@partial(jax.jit, static_argnames=("self",))
	def update(
		self,
		population: Population,
		emitter_state: EmitterState | None,
		random_key: RNGKey,
	) -> tuple[Population, EmitterState | None, Metrics, RNGKey]:
		"""
		Performs one iteration of the MAP-Elites algorithm.
		1. A batch of genotypes is sampled in the population and the genotypes
			are copied.
		2. The copies are mutated and crossed-over
		3. The obtained offsprings are scored and then added to the population.


		Args:
			population: the MAP-Elites population
			emitter_state: state of the emitter
			random_key: a jax PRNG random key

		Returns:
			the updated MAP-Elites population
			the updated (if needed) emitter state
			metrics about the updated population
			a new jax PRNG key
		"""
		# generate offsprings with the emitter
		genotypes, extra_info, random_key = self._emitter.emit(
			population, emitter_state, random_key
		)

		# scores the offsprings
		fitnesses, descriptors, extra_scores, random_key = self._scoring_function(
			genotypes, random_key
		)

		# add genotypes in the population
		population = population.add(genotypes, descriptors, fitnesses, extra_scores)

		# update emitter state after scoring is made
		emitter_state = self._emitter.state_update(
			emitter_state=emitter_state,
			repertoire=population,
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores={**extra_scores, **extra_info},
		)
		# update the metrics
		metrics = self._metrics_function(population)

		return population, emitter_state, metrics, random_key

	@partial(jax.jit, static_argnames=("self",))
	def scan_update(
		self,
		carry: tuple[Population, EmitterState | None, RNGKey],
		unused: Any,
	) -> tuple[tuple[Population, EmitterState | None, RNGKey], Metrics]:
		"""Rewrites the update function in a way that makes it compatible with the
		jax.lax.scan primitive.

		Args:
			carry: a tuple containing the population, the emitter state and a
				random key.
			unused: unused element, necessary to respect jax.lax.scan API.

		Returns:
			The updated population and emitter state, with a new random key and metrics.
		"""
		population, emitter_state, random_key = carry
		(
			population,
			emitter_state,
			metrics,
			random_key,
		) = self.update(
			population,
			emitter_state,
			random_key,
		)

		return (population, emitter_state, random_key), metrics
