import math
import numpy as np
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool
import process_bank
import process_freddie
import os

class MCMCSampler(object):
	def __init__(self, log_f, log_g, g_sample, x0, iterations):
		"""
			Initialize a Metropolis-Hastings sampler for a target distribution P

			log_f: the log of a target distribution (or function proportional to the target distribution) that we will use to calculate acceptance ratios
			g: the log of a function that samples from the marginal probability density that defines the transition probabilities P(x|y) between samples, returns p(x|y) when given inputs x, y
			g_sample: a function that takes in a state x and returns a randomly sampled state in P(*|x)
			x0: the starting sample 
			iterations: number of iterations to ruh MH for
		"""
		self.log_distribution_fn = log_f
		self.log_transition_probabilities = log_g
		self.get_transition_sample = g_sample
		self.state = x0
		self.iterations = iterations

		self.saved_states = [x0]

		self.step_count = 0

	def sample(self):
		raise NotImplementedError

	def calculate_acceptance_ratio(self, proposal_state):
		raise NotImplementedError

	def transition_step(self, candidate_state, acceptance_ratio):
		raise NotImplementedError

class MHSampler(MCMCSampler):
	def sample(self):
		for i in range(self.iterations):
			if i % 1000 == 0:
				print("iteration {}".format(i))
			candidate_state = self.get_transition_sample(self.state)
			acceptance = self.calculate_acceptance_ratio(candidate_state)
			new_state = self.transition_step(candidate_state, acceptance)
			self.saved_states.append(new_state)
			self.state = new_state

			self.step_count += 1

	def calculate_acceptance_ratio(self, proposal_state):
		log = self.log_distribution_fn(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - self.log_distribution_fn(self.state) - self.log_transition_probabilities(proposal_state, self.state)
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		return min(1, acceptance_ratio)

	def transition_step(self, candidate_state, acceptance_ratio):
		u = np.random.uniform()
		return self.state if u > acceptance_ratio else candidate_state

	def get_saved_states(self):
		return self.saved_states

class GibbsSampler(MHSampler):
	def calculate_acceptance_ratio(self, proposal_state):
		return 1.0

class ConsensusMHSampler(MCMCSampler):
	def __init__(self, log_f, log_g, g_sample, x0, iterations, shards=1):
		super(ConsensusMHSampler, self).__init__(log_f, log_g, g_sample, x0, iterations)
		self.shards = shards

		assert len(self.log_distribution_fn) == self.shards
		self.log_fn_dict = {} # for pickling purposes
		for i in range(self.shards):
			self.log_fn_dict[i] = self.log_distribution_fn[i]

		self.pool = Pool(processes=self.shards)

	def sample(self):
		for i in range(self.iterations):
			if i % 1000 == 0:
				print("iteration {}".format(i))
			map_results = self.pool.map(self.map_sample, range(self.shards))
			new_state = self.reduce_sample(map_results)
			self.saved_states.append(new_state)
			self.state = new_state

			self.step_count += 1

	def map_sample(self, index):
		candidate_state = self.get_transition_sample(self.state)
		acceptance = self.calculate_acceptance_ratio(candidate_state, index)
		sample = self.transition_step(candidate_state, acceptance)

		sample_variance = self.calulate_sample_variance(sample)

		return (sample, 1. / sample_variance)

	def reduce_sample(self, results):
		'''
			results is a list of (sample, weight) tuples
		'''

		th_g_num = 0
		th_g_den = 0
		for (sample, weight) in results:
			th_g_num += sample * float(weight)
			th_g_den += float(weight)

		return th_g_num / th_g_den

	def calculate_acceptance_ratio(self, proposal_state, index=0):
		log_fn = self.log_fn_dict[index]
		log = log_fn(proposal_state) + self.log_transition_probabilities(self.state, proposal_state) - log_fn(self.state) - self.log_transition_probabilities(proposal_state, self.state)
		if log > 0:
			acceptance_ratio = 1
		elif log < -20:
			acceptance_ratio = 0
		else:
			acceptance_ratio = np.exp(log)
		return min(1, acceptance_ratio)

	def calculate_sample_variance(sample):
		total_samples = self.get_saved_states() + [sample]
		return np.var(np.array(total_samples), axis=0)

	def transition_step(self, candidate_state, acceptance_ratio):
		u = np.random.uniform()
		return self.state if u > acceptance_ratio else candidate_state

	def get_saved_states(self):
		return self.saved_states


# using main MH on the bank data
def main(dataset):
	if dataset == "bank_small":
		# N=100000 takes 1-2 min
		proposal_variance, burnin, N = .005, 10000, 100000
		feature_array, output_vector = process_bank.create_arrays('bank-additional/bank-additional.csv')
		x0 = [0.25, 3, 3, 5.8, -2.8, -3.7, -3.1, -3.3, -3.3, -3, -2.5, -3.1, -2.9, -2.8, 0.7, 0.7, 1, 1]
	elif dataset == "bank_large":
		# N=20000 takes 1-2 min, N=4000000 takes ~5 hours
		proposal_variance, burnin, N = .0005, 1000, 20000
		feature_array, output_vector = process_bank.create_arrays('bank-additional/bank-additional-full.csv')
		x0 = [0.25, 3, 3, 5.8, -2.8, -3.7, -3.1, -3.3, -3.3, -3, -2.5, -3.1, -2.9, -2.8, 0.7, 0.7, 1, 1]
	elif dataset == "freddie_mac":
		# N=2000 takes about 7 min, N=100000 takes ~6 hours
		proposal_variance, burnin, N = 2e-6, 1000, 20000
		feature_array, output_vector = process_freddie.create_arrays()
		x0 = [0, 0, 0, 0, 0, 0, 0]
	else:
		assert False

	num_features = feature_array.shape[1]
	prior_mean, prior_variance = np.zeros(num_features), 100

	# x0 is currently hard-coded in to avoid burn-in time
	# x0 = np.random.multivariate_normal(prior_mean, np.identity(num_features) * prior_variance)

	def log_multivariate_gaussian_pdf(x, y, variance):
		# assuming the covariance matrix is identity * variance
		return (-0.5 * (x - y).T.dot(np.identity(num_features) / variance).dot(x - y)) - (num_features / 2) * np.log(2 * np.pi * variance)

	def log_f_distribution_fn(val):
		# calculated based on the formula in section 3.1 of the paper
		theta = feature_array.dot(val)
		p_data = np.where(output_vector, theta - np.log(1 + np.exp(theta)), - np.log(1 + np.exp(theta)))
		prior = log_multivariate_gaussian_pdf(prior_mean, val, prior_variance)
		return np.sum(p_data) + prior

	def log_g_transition_prob(data, given):
		return log_multivariate_gaussian_pdf(data, given, proposal_variance)

	def g_sample(val):
		return np.random.multivariate_normal(val, np.identity(num_features) * proposal_variance)

	num_points = feature_array.shape[0]
	split_indices = []
	for i in range(1, 4):
		split_indices.append(int(math.floor(0.25 * i * num_points)))
	split_feature_array = np.split(feature_array, split_indices)

	log_fns = []
	for features in split_feature_array:
		def log_f_split_distribution_fn(val):
			theta = features.dot(val)
			p_data = np.where(output_vector, theta - np.log(1 + np.exp(theta)), - np.log(1 + np.exp(theta)))
			prior = log_multivariate_gaussian_pdf(prior_mean, val, prior_variance)
			return np.sum(p_data) + prior
		log_fns.append(log_f_split_distribution_fn)

	sampler = ConsensusMHSampler(log_fns, log_g_transition_prob, g_sample, x0, N, shards=4)
	sampler.sample()

	samples = sampler.get_saved_states()
	samples = np.array(samples[burnin:])

	for i in range(num_features):
		plot_tracking(samples, i, "plots/{}_100000".format(dataset))

def plot_tracking(samples, index, directory_path):
	plt.figure(1, figsize=(5, 10))

	plt.subplot(211)
	plt.title(r"PDF for Weight Parameter $\beta_{%s}$" % index)
	plt.hist(samples[:, index], 60, density=True)
	plt.xlabel("Value")
	plt.ylabel("Normalized Probability")

	plt.subplot(212)
	factor = int(math.ceil(len(samples) / 2000.)) # only plot 2000 points to make the plot look nicer
	subsamples = samples[::factor, index]
	plt.plot(subsamples, factor * np.arange(len(subsamples)), '.')
	plt.title(r"Distribution over Iterations of Weight Parameter $\beta_{%s}$" % index)
	plt.xlabel("Value")
	plt.ylabel("Sample Iteration")

	if not os.path.exists(directory_path):
		os.makedirs(directory_path)
	plt.savefig("%s/%s_consensus.png" % (directory_path, index), bbox_inches='tight')
	plt.clf()

if __name__ == '__main__':
	main("freddie_mac")
