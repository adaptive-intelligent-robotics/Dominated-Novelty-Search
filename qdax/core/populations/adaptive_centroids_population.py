from __future__ import annotations

from functools import partial

import flax.struct
import jax
import jax.numpy as jnp

from qdax.core.populations.grid_population import GridPopulation
from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def dist(x, y):
	return jnp.linalg.norm(x - y)


dist_v = jax.vmap(dist, in_axes=(None, 0))
dist_vv = jax.vmap(dist_v, in_axes=(0, None))


class AdaptiveCentroidsPopulation(flax.struct.PyTreeNode):
	fitnesses_in_cells: Fitness
	genotypes_in_cells: Genotype
	descriptors_in_cells: Descriptor

	# Instead of vectors, here the centroids are full solutions (with genotypes, fitnesses, descriptors, and extra scores)
	# So centroids are stored in a separate population, called centroids_population
	centroids_population: GridPopulation

	# bias for the fitness of the centroids, the higher the bias, the denser are the areas of high-fitness solutions
	# this parameter is optional, and has no effect if equal to 0 (default)
	bias_fitness: float = flax.struct.field(pytree_node=False, default=0.0)

	@property
	def centroids(self) -> Centroid:
		return self.centroids_population.centroids

	@property
	def descriptors(self) -> Descriptor:
		return jnp.concatenate(
			[self.descriptors_in_cells, self.centroids_population.descriptors], axis=0
		)

	@property
	def fitnesses(self) -> Fitness:
		return jnp.concatenate(
			[self.fitnesses_in_cells, self.centroids_population.fitnesses], axis=0
		)

	@property
	def genotypes(self) -> Genotype:
		return jax.tree.map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			self.genotypes_in_cells,
			self.centroids_population.genotypes,
		)
	
	@property
	def only_in_cells_genotypes(self) -> Genotype:
		return self.genotypes_in_cells
	
	@property
	def only_in_cells_fitnesses(self) -> Fitness:
		return self.fitnesses_in_cells
	
	@property
	def only_in_cells_descriptors(self) -> Descriptor:
		return self.descriptors_in_cells
	

	def get_distance_weights(self, fitnesses: Fitness) -> jnp.ndarray:
		"""Get the weights for the distance computation.

		Args:
		    fitnesses: the fitnesses of the solutions in the population

		Returns:
		    the weights for the distance computation
		"""
		return jnp.exp(self.bias_fitness * fitnesses)

	def _replace_centroids_too_close(
		self,
		old_centroids_population: GridPopulation,
		new_centroids_population: GridPopulation,
	) -> GridPopulation:
		"""
		Merge the new centroids with the old ones. Removing the ones that are too close to each other.
		As each centroid is associated to a solution, it has also a descriptor (equal to the centroid)
		as well as a genotype, a fitness and extra scores.

		:param new_centroids_population: the population whose descriptors are the new centroids to be added to the population
		:return: the updated population of centroids with maximized diversity
		"""
		num_new_centroids = new_centroids_population.centroids.shape[0]

		# Sort the added individuals based on their closest distance to the current centroids.
		# This way, the furthest individuals are added in priority.
		dist_new_old = dist_vv(
			new_centroids_population.centroids, old_centroids_population.centroids
		)
		min_dist_old_new = jnp.min(dist_new_old, axis=1)
		sorted_indices = jnp.argsort(min_dist_old_new, descending=True)
		new_centroids_population = jax.tree.map(
			lambda x: x[sorted_indices], new_centroids_population
		)

		# concatenate the old and new centroids
		all_centroids_population: GridPopulation = jax.tree_map(
			lambda x, y: jnp.concatenate([x, y], axis=0),
			old_centroids_population,
			new_centroids_population,
		)

		# compute the distance matrix between all centroids
		dist_mat = dist_vv(
			all_centroids_population.centroids, all_centroids_population.centroids
		)

		# apply a bias to the fitness of the centroids, the higher the bias, the more distant the centroids are considered
		dist_mat = jax.vmap(lambda x, y: x * y, in_axes=(0, None))(
			dist_mat, self.get_distance_weights(all_centroids_population.fitnesses)
		)

		# mask upper triangle matrix to only calculate distances in one direction
		# this way, the original centroids think they're infinitely far from the new ones,
		# so they have fewer chances to be removed.
		mask = jnp.tril(jnp.ones(shape=dist_mat.shape), k=-1)
		dist_mat = jnp.where(mask, dist_mat, jnp.inf)

		# compute minimum distance of each centroid to the others
		min_dists = jnp.min(dist_mat, axis=1)
		highest_indices = jnp.argpartition(min_dists, kth=num_new_centroids)[
			num_new_centroids:
		]

		# keep only the centroids that are not too close to each other
		final_centroids_population = jax.tree_map(
			lambda x: x[highest_indices],
			all_centroids_population,
		)

		return final_centroids_population

	def _update_centroids_individuals(
		self,
		batch_centroid_individuals: GridPopulation,
	) -> GridPopulation:
		# replace old centroids with new ones if they are more distant
		result_population = self._replace_centroids_too_close(
			old_centroids_population=self.centroids_population,
			new_centroids_population=batch_centroid_individuals,
		)

		# Here perform additional steps if needed
		# ...

		return result_population

	@classmethod
	def init(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		extra_scores: ExtraScores | None = None,
	) -> AdaptiveCentroidsPopulation:
		"""
		Initialize a population with an initial population of genotypes.
		"""

		# retrieve one genotype from the population
		return cls.init_default(
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			extra_scores=extra_scores,
		)

	@classmethod
	def init_default(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		extra_scores: ExtraScores | None = None,
	) -> AdaptiveCentroidsPopulation:
		"""
		Initialize a population with an initial population of
		genotypes. In this population, the centroids are also solutions,
		so instead of vectors, the centroids correspond to full solutions
		(with genotypes, fitnesses, descriptors, and extra scores).
		So centroids are stored in a separate population, called centroids_population.
		In this centroids_population, the centroids coincide with the descriptors.
		"""

		centroids_population = GridPopulation(
			genotypes=genotypes,
			fitnesses=fitnesses,
			descriptors=descriptors,
			centroids=descriptors,  # centroids are the same as descriptors in the centroids population
		)

		return cls(
			genotypes_in_cells=genotypes,
			fitnesses_in_cells=fitnesses,
			descriptors_in_cells=descriptors,
			centroids_population=centroids_population,
		)

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

		population_empty = self.fitnesses_in_cells == -jnp.inf
		p = (1.0 - population_empty) / jnp.sum(1.0 - population_empty)

		random_key, subkey = jax.random.split(random_key)
		samples = jax.tree.map(
			lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
			self.genotypes_in_cells,
		)

		return samples, random_key

	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_extra_scores: ExtraScores | None = None,
	) -> AdaptiveCentroidsPopulation:
		if batch_of_extra_scores is None:
			batch_of_extra_scores = {}

		sum_desc_coords = jnp.sum(batch_of_descriptors, axis=1).reshape(
			batch_of_fitnesses.shape
		)

		batch_of_fitnesses = jnp.where(
			jnp.isnan(sum_desc_coords), -1 * jnp.inf, batch_of_fitnesses
		)
		batch_of_descriptors = jnp.where(
			jnp.isnan(batch_of_descriptors), 0.0, batch_of_descriptors
		)

		batch_centroid_individuals = GridPopulation(
			genotypes=batch_of_genotypes,
			fitnesses=batch_of_fitnesses,
			descriptors=batch_of_descriptors,
			centroids=batch_of_descriptors,
			# centroids are the same as descriptors (as this is a population of centroids, i.e. centroids correspond to actual solutions)
		)

		# update centroids
		new_population_centroids_individuals = self._update_centroids_individuals(
			batch_centroid_individuals=batch_centroid_individuals,
		)

		# Now that centroids have been updated, we reset the population with the new centroids
		new_population = GridPopulation.init(
			self.genotypes_in_cells,
			self.fitnesses_in_cells,
			self.descriptors_in_cells,
			new_population_centroids_individuals.centroids,
		)

		# add the new individuals to the population
		new_population = new_population.add(
			batch_of_genotypes,
			batch_of_descriptors,
			batch_of_fitnesses,
			batch_of_extra_scores,
		)

		# add the centroids individuals to the population (just in case there are still empty cells)
		new_population = new_population.add(
			new_population_centroids_individuals.genotypes,
			new_population_centroids_individuals.centroids,  # descriptors = centroids
			new_population_centroids_individuals.fitnesses,
		)

		return self.replace(
			genotypes_in_cells=new_population.genotypes,
			fitnesses_in_cells=new_population.fitnesses,
			descriptors_in_cells=new_population.descriptors,
			centroids_population=new_population_centroids_individuals,
		)
