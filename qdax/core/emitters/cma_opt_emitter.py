from __future__ import annotations

import jax.numpy as jnp

from qdax.core.emitters.cma_emitter import CMAEmitter, CMAEmitterState
from qdax.core.populations.grid_population import GridPopulation
from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype


class CMAOptimizingEmitter(CMAEmitter):
	def _ranking_criteria(
		self,
		emitter_state: CMAEmitterState,
		repertoire: GridPopulation,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		extra_scores: ExtraScores | None,
		improvements: jnp.ndarray,
	) -> jnp.ndarray:
		"""Defines how the genotypes should be sorted. Impacts the update
		of the CMAES state. In the end, this defines the type of CMAES emitter
		used (optimizing, random direction or improvement).

		Args:
		    emitter_state: current state of the emitter.
		    repertoire: latest repertoire of genotypes.
		    genotypes: emitted genotypes.
		    fitnesses: corresponding fitnesses.
		    descriptors: corresponding fitnesses.
		    extra_scores: corresponding extra scores.
		    improvements: improvments of the emitted genotypes. This corresponds
		        to the difference between their fitness and the fitness of the
		        individual occupying the cell of corresponding fitness.

		Returns:
		    The values to take into account in order to rank the emitted genotypes.
		    Here, it is the fitness of the genotype.
		"""

		return fitnesses
