from __future__ import annotations

from functools import partial

import flax.struct
import jax
import jax.numpy as jnp

from qdax.custom_types import Centroid, Descriptor, Fitness, Genotype, ExtraScores, RNGKey, Observation


@partial(jax.jit, static_argnames=("k_nn",))
def get_cells_indices(
	batch_of_descriptors: Descriptor, centroids: Centroid, k_nn: int
) -> tuple[jnp.ndarray, jnp.ndarray]:
	"""Returns the array of cells indices for a batch of descriptors
	given the centroids of the grid.

	Args:
	    batch_of_descriptors: a batch of descriptors
	        of shape (batch_size, num_descriptors)
	    centroids: centroids array of shape (num_centroids, num_descriptors)

	Returns:
	    the indices of the centroids corresponding to each vector of descriptors
	        in the batch with shape (batch_size,)
	"""

	def _get_cells_indices(
		_descriptors: jnp.ndarray,
		_centroids: jnp.ndarray,
		_k_nn: int,
	) -> tuple[jnp.ndarray, jnp.ndarray]:
		"""Inner function.

		descriptors of shape (1, num_descriptors)
		centroids of shape (num_centroids, num_descriptors)
		"""

		distances = jax.vmap(jnp.linalg.norm)(_descriptors - _centroids)

		# Negating distances because we want the smallest ones
		min_dist, min_args = jax.lax.top_k(-1 * distances, _k_nn)

		return min_args, -1 * min_dist

	func = jax.vmap(
		_get_cells_indices,
		in_axes=(
			0,
			None,
			None,
		),
	)

	return func(batch_of_descriptors, centroids, k_nn)


@jax.jit
def intra_batch_comp(
	normed: jnp.ndarray,
	current_index: jnp.ndarray,
	normed_all: jnp.ndarray,
	eval_scores: jnp.ndarray,
	l_value: jnp.ndarray,
) -> jnp.ndarray:
	"""Function to know if an individual should be kept or not."""

	# Check for individuals that are Nans, we remove them at the end
	not_existent = jnp.where((jnp.isnan(normed)).any(), True, False)

	# Fill in Nans to do computations
	normed = jnp.where(jnp.isnan(normed), jnp.full(normed.shape[-1], jnp.inf), normed)
	eval_scores = jnp.where(
		jnp.isinf(eval_scores), jnp.full(eval_scores.shape[-1], jnp.nan), eval_scores
	)

	# If we do not use a fitness (i.e same fitness everywhere), we create a virtual
	# fitness function to add individuals with the same bd
	additional_score = jnp.where(
		jnp.nanmax(eval_scores) == jnp.nanmin(eval_scores), 1.0, 0.0
	)
	additional_scores = jnp.linspace(0.0, additional_score, num=eval_scores.shape[0])

	# Add scores to empty individuals
	eval_scores = jnp.where(
		jnp.isnan(eval_scores), jnp.full(eval_scores.shape[0], -jnp.inf), eval_scores
	)
	# Virtual eval_scores
	eval_scores = eval_scores + additional_scores

	# For each point we check what other points are the closest ones.
	knn_relevant_scores, knn_relevant_indices = jax.lax.top_k(
		-1 * jax.vmap(jnp.linalg.norm)(normed - normed_all), eval_scores.shape[0]
	)
	# We negated the scores to use top_k so we reverse it.
	knn_relevant_scores = knn_relevant_scores * -1

	# Check if the individual is close enough to compare (under l-value)
	fitness = jnp.where(jnp.squeeze(knn_relevant_scores < l_value), True, False)

	# We want to eliminate the same individual (distance 0)
	fitness = jnp.where(knn_relevant_indices == current_index, False, fitness)
	current_fitness = jnp.squeeze(
		eval_scores.at[knn_relevant_indices.at[0].get()].get()
	)

	# Is the fitness of the other individual higher?
	# If both are True then we discard the current individual since this individual
	# would be replaced by the better one.
	discard_indiv = jnp.logical_and(
		jnp.where(
			eval_scores.at[knn_relevant_indices].get() > current_fitness, True, False
		),
		fitness,
	).any()

	# Discard Individuals with Nans as their BD (mainly for the readdition where we
	# have NaN bds)
	discard_indiv = jnp.logical_or(discard_indiv, not_existent)

	# Negate to know if we keep the individual
	return jnp.logical_not(discard_indiv)


class ThresholdPopulation(flax.struct.PyTreeNode):
	"""
	Class for the threshold population.

	Args:
	    genotypes: Genotypes of the individuals in the population.
	    fitnesses: Fitnesses of the individuals in the population.
	    descriptors: Descriptors of the individuals in the population.
	    centroids: Centroids of the tesselation.
	    observations: Observations that the genotype gathered in the environment.
	"""

	genotypes: Genotype
	fitnesses: Fitness
	descriptors: Descriptor
	observations: Observation
	l_value: jnp.ndarray

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

	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		extra_scores: ExtraScores,
	) -> ThresholdPopulation:
		"""Adds a batch of genotypes to the population.

		Args:
			batch_of_genotypes: genotypes of the individuals to be considered
				for addition in the population.
			batch_of_descriptors: associated descriptors.
			batch_of_fitnesses: associated fitness.
			extra_scores: associated extra scores.

		Returns:
		    The updated population.
		"""

		# We need to replace all the descriptors that are not filled with jnp inf
		filtered_descriptors = jnp.where(
			jnp.expand_dims((self.fitnesses == -jnp.inf), axis=-1),
			jnp.full(self.descriptors.shape[-1], fill_value=jnp.inf),
			self.descriptors,
		)

		batch_of_indices, batch_of_distances = get_cells_indices(
			batch_of_descriptors, filtered_descriptors, 2
		)

		# Save the second-nearest neighbours to check a condition
		second_neighbours = batch_of_distances.at[..., 1].get()

		# Keep the Nearest neighbours
		batch_of_indices = batch_of_indices.at[..., 0].get()

		# Keep the Nearest neighbours
		batch_of_distances = batch_of_distances.at[..., 0].get()

		# We remove individuals that are too close to the second nn.
		# This avoids having clusters of individuals after adding them.
		not_novel_enough = jnp.where(
			jnp.squeeze(second_neighbours <= self.l_value), True, False
		)

		#if extra_scores has last valid observations, use them
		if "last_valid_observations" in extra_scores:
			batch_of_observations = extra_scores["last_valid_observations"]
		else:
			batch_of_observations = jnp.zeros_like(self.observations)

		# batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
		batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)
		batch_of_observations = jnp.expand_dims(batch_of_observations, axis=-1)

		# TODO: Doesn't Work if population is full. Need to use the closest individuals
		# in that case.
		empty_indexes = jnp.squeeze(
			jnp.nonzero(
				jnp.where(jnp.isinf(self.fitnesses), 1, 0),
				size=batch_of_indices.shape[0],
				fill_value=-1,
			)[0]
		)
		batch_of_indices = jnp.where(
			jnp.squeeze(batch_of_distances <= self.l_value),
			jnp.squeeze(batch_of_indices),
			-1,
		)

		# We get all the indices of the empty bds first and then the filled ones
		# (because of -1)
		sorted_bds = jax.lax.top_k(
			-1 * batch_of_indices.squeeze(), batch_of_indices.shape[0]
		)[1]
		batch_of_indices = jnp.where(
			jnp.squeeze(batch_of_distances.at[sorted_bds].get() <= self.l_value),
			batch_of_indices.at[sorted_bds].get(),
			empty_indexes,
		)

		batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)

		# ReIndexing of all the inputs to the correct sorted way
		batch_of_descriptors = batch_of_descriptors.at[sorted_bds].get()
		batch_of_genotypes = jax.tree_map(
			lambda x: x.at[sorted_bds].get(), batch_of_genotypes
		)
		batch_of_fitnesses = batch_of_fitnesses.at[sorted_bds].get()
		batch_of_observations = batch_of_observations.at[sorted_bds].get()
		not_novel_enough = not_novel_enough.at[sorted_bds].get()

		# Check to find Individuals with same BD within the Batch
		keep_indiv = jax.jit(
			jax.vmap(intra_batch_comp, in_axes=(0, 0, None, None, None), out_axes=(0))
		)(
			batch_of_descriptors.squeeze(),
			jnp.arange(
				0, batch_of_descriptors.shape[0], 1
			),  # keep track of where we are in the batch to assure right comparisons
			batch_of_descriptors.squeeze(),
			batch_of_fitnesses.squeeze(),
			self.l_value,
		)

		keep_indiv = jnp.logical_and(keep_indiv, jnp.logical_not(not_novel_enough))

		# get fitness segment max
		best_fitnesses = jax.ops.segment_max(
			batch_of_fitnesses,
			batch_of_indices.astype(jnp.int32).squeeze(),
			num_segments=self.max_size,
		)

		cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

		# put dominated fitness to -jnp.inf
		batch_of_fitnesses = jnp.where(
			batch_of_fitnesses == cond_values,
			batch_of_fitnesses,
			-jnp.inf
		)

		# get addition condition
		grid_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
		current_fitnesses = jnp.take_along_axis(grid_fitnesses, batch_of_indices, 0)
		addition_condition = batch_of_fitnesses > current_fitnesses
		addition_condition = jnp.logical_and(
			addition_condition, jnp.expand_dims(keep_indiv, axis=-1)
		)

		# assign fake position when relevant : num_centroids is out of bounds
		batch_of_indices = jnp.where(
			addition_condition,
			batch_of_indices,
			self.max_size,
		)

		# create new grid
		new_grid_genotypes = jax.tree_map(
			lambda grid_genotypes, new_genotypes: grid_genotypes.at[
				batch_of_indices.squeeze()
			].set(new_genotypes),
			self.genotypes,
			batch_of_genotypes,
		)

		# compute new fitness and descriptors
		new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze()].set(
			batch_of_fitnesses.squeeze()
		)
		new_descriptors = self.descriptors.at[batch_of_indices.squeeze()].set(
			batch_of_descriptors.squeeze()
		)

		new_observations = self.observations.at[batch_of_indices.squeeze()].set(
			batch_of_observations.squeeze()
		)

		return ThresholdPopulation(
			genotypes=new_grid_genotypes,
			fitnesses=new_fitnesses.squeeze(),
			descriptors=new_descriptors.squeeze(),
			observations=new_observations.squeeze(),
			l_value=self.l_value,
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

		random_key, sub_key = jax.random.split(random_key)
		grid_empty = self.fitnesses == -jnp.inf
		p = (1.0 - grid_empty) / jnp.sum(1.0 - grid_empty)

		samples = jax.tree_map(
			lambda x: jax.random.choice(sub_key, x, shape=(num_samples,), p=p),
			self.genotypes,
		)

		return samples, random_key

	@classmethod
	def init(
		cls,
		genotypes: Genotype,
		fitnesses: Fitness,
		descriptors: Descriptor,
		observations: Observation,
		l_value: jnp.ndarray,
		max_size: int,
	) -> ThresholdPopulation:
		"""Initialize a threshold population with an initial population of genotypes.
		Requires the definition of centroids that can be computed with any method
		such as CVT or Euclidean mapping.

		Args:
		    genotypes: initial genotypes, pytree in which leaves
		        have shape (batch_size, num_features)
		    fitnesses: fitness of the initial genotypes of shape (batch_size,)
		    descriptors: descriptors of the initial genotypes
		        of shape (batch_size, num_descriptors)
		    observations: observations experienced in the evaluation task.
		    l_value: threshold distance of the population.
		    max_size: maximal size of the container

		Returns:
		    an initialized threshold population.
		"""

		# Initialize grid with default values
		default_fitnesses = -jnp.inf * jnp.ones(shape=max_size)
		default_genotypes = jax.tree_map(
			lambda x: jnp.full(shape=(max_size,) + x.shape[1:], fill_value=jnp.nan),
			genotypes,
		)
		default_descriptors = jnp.zeros(shape=(max_size, descriptors.shape[-1]))

		default_observations = jnp.full(
			shape=(max_size,) + observations.shape[1:], fill_value=jnp.nan
		)

		population = ThresholdPopulation(
			genotypes=default_genotypes,
			fitnesses=default_fitnesses,
			descriptors=default_descriptors,
			observations=default_observations,
			l_value=l_value,
		)

		return population.add(genotypes, descriptors, fitnesses, observations)
