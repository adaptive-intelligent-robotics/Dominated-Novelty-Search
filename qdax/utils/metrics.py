"""Defines functions to retrieve metrics from training processes."""

from __future__ import annotations

import csv

from jax import numpy as jnp

from qdax.core.populations.ga_population import GAPopulation
from qdax.core.populations.grid_population import GridPopulation
from qdax.custom_types import Metrics


class CSVLogger:
	"""Logger to save metrics of an experiment in a csv file
	during the training process.
	"""

	def __init__(self, filename: str, header: list) -> None:
		"""Create the csv logger, create a file and write the
		header.

		Args:
			filename: path to which the file will be saved.
			header: header of the csv file.
		"""
		self._filename = filename
		self._header = header
		with open(self._filename, "w") as file:
			writer = csv.DictWriter(file, fieldnames=self._header)
			# write the header
			writer.writeheader()

	def log(self, metrics: dict[str, float]) -> None:
		"""Log new metrics to the csv file.

		Args:
			metrics: A dictionary containing the metrics that
				need to be saved.
		"""
		with open(self._filename, "a") as file:
			writer = csv.DictWriter(file, fieldnames=self._header)
			# write new metrics in a raw
			writer.writerow(metrics)


def default_ga_metrics(
	repertoire: GAPopulation,
) -> Metrics:
	"""Compute the usual GA metrics that one can retrieve
	from a GA repertoire.

	Args:
		repertoire: a GA repertoire

	Returns:
		a dictionary containing the max fitness of the
			repertoire.
	"""

	# get metrics
	max_fitness = jnp.max(repertoire.fitnesses, axis=0)

	return {
		"max_fitness": max_fitness,
	}


def default_qd_metrics(repertoire: GridPopulation, qd_offset: float) -> Metrics:
	"""Compute the usual QD metrics that one can retrieve
	from a MAP Elites repertoire.

	Args:
		repertoire: a MAP-Elites repertoire
		qd_offset: an offset used to ensure that the QD score
			will be positive and increasing with the number
			of individuals.

	Returns:
		a dictionary containing the QD score (sum of fitnesses
			modified to be all positive), the max fitness of the
			repertoire, the coverage (number of niche filled in
			the repertoire).
	"""
	repertoire_empty = repertoire.fitnesses == -jnp.inf

	qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
	qd_score += qd_offset * jnp.sum(1.0 - repertoire_empty)

	coverage = jnp.mean(1.0 - repertoire_empty)

	max_fitness = jnp.max(repertoire.fitnesses)

	mean_fitness = jnp.mean(repertoire.fitnesses, where=~repertoire_empty)

	return {
		"qd_score": qd_score,
		"coverage": coverage,
		"max_fitness": max_fitness,
		"mean_fitness": mean_fitness,
	}