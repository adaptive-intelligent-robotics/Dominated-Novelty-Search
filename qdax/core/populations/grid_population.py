"""Grid population class for Quality-Diversity algorithms."""

from __future__ import annotations

import warnings
from functools import partial

import flax
import jax
import jax.numpy as jnp
from numpy.random import RandomState
from sklearn.cluster import KMeans

from qdax.custom_types import Centroid, Descriptor, ExtraScores, Fitness, Genotype, RNGKey


def compute_cvt_centroids(
	num_descriptors: int,
	num_init_cvt_samples: int,
	num_centroids: int,
	minval: float | list[float],
	maxval: float | list[float],
	random_key: RNGKey,
) -> tuple[jnp.ndarray, RNGKey]:
	"""Compute centroids for CVT tessellation.

	Args:
	    num_descriptors: number of scalar descriptors
	    num_init_cvt_samples: number of sampled point to be sued for clustering to
	        determine the centroids. The larger the number of centroids and the
	        number of descriptors, the higher this value must be (e.g. 100000 for
	        1024 centroids and 100 descriptors).
	    num_centroids: number of centroids
	    minval: minimum descriptors value
	    maxval: maximum descriptors value
	    random_key: a jax PRNG random key

	Returns:
	    the centroids with shape (num_centroids, num_descriptors)
	    random_key: an updated jax PRNG random key
	"""
	minval = jnp.array(minval)
	maxval = jnp.array(maxval)

	# assume here all values are in [0, 1] and rescale later
	random_key, subkey = jax.random.split(random_key)
	x = jax.random.uniform(key=subkey, shape=(num_init_cvt_samples, num_descriptors))

	# compute k means
	random_key, subkey = jax.random.split(random_key)
	k_means = KMeans(
		init="k-means++",
		n_clusters=num_centroids,
		n_init=1,
		random_state=RandomState(jax.random.key_data(subkey)),
	)
	k_means.fit(x)
	centroids = k_means.cluster_centers_
	# rescale now
	return jnp.asarray(centroids) * (maxval - minval) + minval, random_key


def compute_euclidean_centroids(
	grid_shape: tuple[int, ...],
	minval: float | list[float],
	maxval: float | list[float],
) -> jnp.ndarray:
	"""Compute centroids for square Euclidean tessellation.

	Args:
	    grid_shape: number of centroids per BD dimension
	    minval: minimum descriptors value
	    maxval: maximum descriptors value

	Returns:
	    the centroids with shape (num_centroids, num_descriptors)
	"""
	# get number of descriptors
	num_descriptors = len(grid_shape)

	# prepare list of linspaces
	linspace_list = []
	for num_centroids_in_dim in grid_shape:
		offset = 1 / (2 * num_centroids_in_dim)
		linspace = jnp.linspace(offset, 1.0 - offset, num_centroids_in_dim)
		linspace_list.append(linspace)

	meshes = jnp.meshgrid(*linspace_list, sparse=False)

	# create centroids
	centroids = jnp.stack(
		[jnp.ravel(meshes[i]) for i in range(num_descriptors)], axis=-1
	)
	minval = jnp.array(minval)
	maxval = jnp.array(maxval)
	return jnp.asarray(centroids) * (maxval - minval) + minval


def get_cells_indices(
	batch_of_descriptors: jnp.ndarray, centroids: jnp.ndarray
) -> jnp.ndarray:
	"""
	Returns the array of cells indices for a batch of descriptors
	given the centroids of the population.

	Args:
	    batch_of_descriptors: a batch of descriptors
	        of shape (batch_size, num_descriptors)
	    centroids: centroids array of shape (num_centroids, num_descriptors)

	Returns:
	    the indices of the centroids corresponding to each vector of descriptors
	        in the batch with shape (batch_size,)
	"""

	def _get_cells_indices(
		descriptors: jnp.ndarray, centroids: jnp.ndarray
	) -> jnp.ndarray:
		"""Set_of_descriptors of shape (1, num_descriptors)
		centroids of shape (num_centroids, num_descriptors)
		"""
		return jnp.argmin(
			jnp.sum(jnp.square(jnp.subtract(descriptors, centroids)), axis=-1)
		)

	func = jax.vmap(lambda x: _get_cells_indices(x, centroids))
	return func(batch_of_descriptors)


class GridPopulation(flax.struct.PyTreeNode):
	"""Class for the population in Map Elites.

	Args:
	    genotypes: Genotypes of the individuals in the population.
	    fitnesses: Fitnesses of the individuals in the population.
	    descriptors: Descriptors of the individuals in the population.
	    centroids: Centroids of the tessellation.
	"""

	genotypes: Genotype
	fitnesses: Fitness
	descriptors: Descriptor
	centroids: Centroid

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
		descriptors: Descriptor,
		centroids: Centroid,
		extra_scores: ExtraScores | None = None,
	) -> GridPopulation:
		"""
		Initialize a grid population with an initial population of genotypes.
		Requires the definition of centroids that can be computed with any method
		such as CVT or Euclidean mapping.

		Args:
		    genotypes: initial genotypes, pytree in which leaves
		        have shape (batch_size, num_features)
		    fitnesses: fitness of the initial genotypes of shape (batch_size,)
		    descriptors: descriptors of the initial genotypes
		        of shape (batch_size, num_descriptors)
		    centroids: tesselation centroids of shape (batch_size, num_descriptors)
		    extra_scores: unused extra_scores of the initial genotypes

		Returns:
		    an initialized grid population
		"""
		warnings.warn(
			(
				"This type of population does not store the extra scores "
				"computed by the scoring function"
			),
			stacklevel=2,
		)

		# retrieve one genotype from the population
		first_genotype = jax.tree.map(lambda x: x[0], genotypes)

		# create a population with default values
		population = cls.init_default(genotype=first_genotype, centroids=centroids)

		# add initial population to the population
		new_population = population.add(genotypes, descriptors, fitnesses, extra_scores)

		return new_population

	@classmethod
	def init_default(
		cls,
		genotype: Genotype,
		centroids: Centroid,
	) -> GridPopulation:
		"""Initialize a grid population with an initial population of
		genotypes. Requires the definition of centroids that can be computed
		with any method such as CVT or Euclidean mapping.

		Args:
		    genotype: the typical genotype that will be stored.
		    centroids: the centroids of the population

		Returns:
		    A population filled with default values.
		"""

		# get number of centroids
		num_centroids = centroids.shape[0]

		# default fitness is -inf
		default_fitnesses = -jnp.inf * jnp.ones(shape=num_centroids)

		# default genotypes is all 0
		default_genotypes = jax.tree.map(
			lambda x: jnp.zeros(shape=(num_centroids,) + x.shape, dtype=x.dtype),
			genotype,
		)

		# default descriptor is all zeros
		default_descriptors = jnp.zeros_like(centroids)

		return cls(
			genotypes=default_genotypes,
			fitnesses=default_fitnesses,
			descriptors=default_descriptors,
			centroids=centroids,
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

		population_empty = self.fitnesses == -jnp.inf
		p = (1.0 - population_empty) / jnp.sum(1.0 - population_empty)

		random_key, subkey = jax.random.split(random_key)
		samples = jax.tree.map(
			lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
			self.genotypes,
		)

		return samples, random_key

	@partial(jax.jit, static_argnames=("num_samples",))
	def sample_with_descs(
		self,
		random_key: RNGKey,
		num_samples: int,
	) -> tuple[Genotype, Descriptor, RNGKey]:
		"""Sample elements in the population.

		Args:
		    random_key: a jax PRNG random key
		    num_samples: the number of elements to be sampled

		Returns:
		    samples: a batch of genotypes sampled in the population
		    random_key: an updated jax PRNG random key
		"""

		population_empty = self.fitnesses == -jnp.inf
		p = (1.0 - population_empty) / jnp.sum(1.0 - population_empty)

		random_key, subkey = jax.random.split(random_key)
		samples = jax.tree.map(
			lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
			self.genotypes,
		)
		descs = jax.tree.map(
			lambda x: jax.random.choice(subkey, x, shape=(num_samples,), p=p),
			self.descriptors,
		)

		return samples, descs, random_key

	@jax.jit
	def add(
		self,
		batch_of_genotypes: Genotype,
		batch_of_descriptors: Descriptor,
		batch_of_fitnesses: Fitness,
		batch_of_extra_scores: ExtraScores | None = None,
	) -> GridPopulation:
		"""
		Add a batch of elements to the population.

		Args:
		    batch_of_genotypes: a batch of genotypes to be added to the population.
		        Similarly to the self.genotypes argument, this is a PyTree in which
		        the leaves have a shape (batch_size, num_features)
		    batch_of_descriptors: an array that contains the descriptors of the
		        aforementioned genotypes. Its shape is (batch_size, num_descriptors)
		    batch_of_fitnesses: an array that contains the fitnesses of the
		        aforementioned genotypes. Its shape is (batch_size,)
		    batch_of_extra_scores: unused tree that contains the extra_scores of
		        aforementioned genotypes.

		Returns:
		    The updated grid population.
		"""

		batch_of_indices = get_cells_indices(batch_of_descriptors, self.centroids)
		batch_of_indices = jnp.expand_dims(batch_of_indices, axis=-1)
		batch_of_fitnesses = jnp.expand_dims(batch_of_fitnesses, axis=-1)

		num_centroids = self.centroids.shape[0]

		# get fitness segment max
		best_fitnesses = jax.ops.segment_max(
			batch_of_fitnesses,
			batch_of_indices.astype(jnp.int32).squeeze(axis=-1),
			num_segments=num_centroids,
		)

		cond_values = jnp.take_along_axis(best_fitnesses, batch_of_indices, 0)

		# put dominated fitness to -jnp.inf
		batch_of_fitnesses = jnp.where(
			batch_of_fitnesses == cond_values, batch_of_fitnesses, -jnp.inf
		)

		# get addition condition
		population_fitnesses = jnp.expand_dims(self.fitnesses, axis=-1)
		current_fitnesses = jnp.take_along_axis(
			population_fitnesses, batch_of_indices, 0
		)
		addition_condition = batch_of_fitnesses > current_fitnesses

		# assign fake position when relevant : num_centroids is out of bound
		batch_of_indices = jnp.where(
			addition_condition, batch_of_indices, num_centroids
		)

		# create new population
		new_population_genotypes = jax.tree.map(
			lambda population_genotypes, new_genotypes: population_genotypes.at[
				batch_of_indices.squeeze(axis=-1)
			].set(new_genotypes),
			self.genotypes,
			batch_of_genotypes,
		)

		# compute new fitness and descriptors
		new_fitnesses = self.fitnesses.at[batch_of_indices.squeeze(axis=-1)].set(
			batch_of_fitnesses.squeeze(axis=-1)
		)
		new_descriptors = self.descriptors.at[batch_of_indices.squeeze(axis=-1)].set(
			batch_of_descriptors
		)

		return GridPopulation(
			genotypes=new_population_genotypes,
			fitnesses=new_fitnesses,
			descriptors=new_descriptors,
			centroids=self.centroids,
		)
