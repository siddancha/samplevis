import tensorflow as tf
import tensorflow_probability as tfp
from easydict import EasyDict as edict
from util import merge_dicts

default_hmc_opts = edict(
    step_size=5e-5,
    num_leapfrog_steps=5,
    num_results=10,
    num_burnin_steps=1000,
    num_steps_between_results=0
)

def hmc(t_image, target_log_prob_fn, hmc_opts):
    hmc_opts = merge_dicts(dict_user=hmc_opts, dict_default=default_hmc_opts)

    # Create state to hold updated `step_size`.
    step_size = tf.get_variable(
        name='step_size',
        initializer=hmc_opts.step_size,
        use_resource=True,
        trainable=False)

    # Initialize the HMC transition kernel.
    hmc_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_log_prob_fn,
        num_leapfrog_steps=hmc_opts.num_leapfrog_steps,
        step_size=step_size,
        step_size_update_fn=tfp.mcmc.make_simple_step_size_update_policy())

    # Run the chain (with burn-in).
    samples, kernel_results = tfp.mcmc.sample_chain(
        num_results=hmc_opts.num_results,
        num_burnin_steps=hmc_opts.num_burnin_steps,
        num_steps_between_results=hmc_opts.num_steps_between_results,
        current_state=t_image,
        kernel=hmc_kernel)

    # Initialize all constructed variables.
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        init_op.run()
        samples_ = sess.run([samples, kernel_results])[0]

    return samples_
