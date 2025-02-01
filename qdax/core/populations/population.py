"""Abstract class for populations."""

from __future__ import annotations

from abc import ABC, abstractmethod

import flax

from qdax.custom_types import Genotype, RNGKey


class Population(flax.struct.PyTreeNode, ABC):
	"""Abstract class for populations."""

	@property
	def max_size(self) -> int:
		"""Gives the max size of the population."""
		pass

	@property
	def size(self) -> int:
		"""Gives the size of the population."""
		pass

	@classmethod
	@abstractmethod
	def init(cls) -> Population:  # noqa: N805
		"""Create a population."""
		pass

	@abstractmethod
	def sample(
		self,
		random_key: RNGKey,
		num_samples: int,
	) -> Genotype:
		"""Sample genotypes from the population.

		Args:
		    random_key: a random key to handle stochasticity.
		    num_samples: the number of genotypes to sample.

		Returns:
		    The sample of genotypes.
		"""
		pass

	@abstractmethod
	def add(self) -> Population:
		"""Adds a batch of genotypes to the population.

		Returns:
		    The udpated population.
		"""
		pass
