import numpy as np
import matplotlib.pyplot as plt

class MHSampler(object):
	def __init__(self, f, g, g_sample, x0, iterations):
		"""
			Initialize a Metropolis-Hastings sampler for a target distribution P

			f: a target distribution (or function proportional to the target distribution) that we will use to calculate acceptance ratios
			g: a function that samples from the marginal probability density that defines the transition probabilities P(x|y) between samples, returns p(x|y) when given inputs x, y
			g_sample: a function that takes in a state x and returns a randomly sampled state in P(*|x)
			x0: the starting sample 
			iterations: number of iterations to ruh MH for
		"""
		self.distribution_fn = f
		self.get_transition_probabilities = g
		self.get_transition_sample = g_sample
		self.state = x0
		self.iterations = iterations

		self.saved_states = [x0]

		self.step_count = 0

	def sample(self):
		for i in range(self.iterations):
			candidate_state = self.get_transition_sample(self.state)
			acceptance = self.calculate_acceptance_ratio(candidate_state)
			new_state = self.transition_step(candidate_state, acceptance)
			self.saved_states.append(new_state)
			self.state = new_state

			self.step_count += 1

	def calculate_acceptance_ratio(self, proposal_state):
		acceptance_ratio = self.distribution_fn(proposal_state) * self.get_transition_probabilities(self.state, proposal_state) / (self.distribution_fn(self.state) * self.get_transition_probabilities(proposal_state, self.state))
		return min(1, acceptance_ratio)

	def transition_step(self, candidate_state, acceptance_ratio):
		u = np.random.uniform()
		return self.state if u > acceptance_ratio else candidate_state

	def get_saved_states(self):
		return self.saved_states

def main():
	# estimate mean given sig = 1
	true_mu, true_sig, N = 0, 1, 100000

	x0 = np.random.uniform(-1.0, 1.0)

	def f_distribution_fn(val):
		return np.exp(-val**2 / 2)/np.sqrt(2 * np.pi)

	def g_transition_prob(data, given):
		# assume we know sig = 1
		return np.exp(-(data - given)**2 / 2)/np.sqrt(2 * np.pi)

	def g_sample(val):
		return np.random.normal(loc = val, scale = true_sig)

	sampler = MHSampler(f_distribution_fn, g_transition_prob, g_sample, x0, N)
	sampler.sample()

	samples = sampler.get_saved_states()

	count, bins, ignored = plt.hist(np.array(samples), 30, normed=True)

	plt.plot(bins, 1/(true_sig * np.sqrt(2 * np.pi)) * np.exp( - (bins - true_mu)**2 / (2 * true_sig**2) ), linewidth=2, color='r')

	plt.show()

if __name__ == '__main__':
	main()