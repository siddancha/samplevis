import tensorflow as tf

from lucid.optvis import param
from lucid.optvis import render as optvis_render

from hmc import hmc
from util import sigmoid, inverse_sigmoid


def render(model, obj, hmc_opts={}, im_size=128):
    """
    Args:
        model: A pre-trained model loaded from Lucid's modelzoo.
        obj: An objective function which defines the sampling probability. From lucid.optvis.objectives.
        hmc_opts: Dictionary of HMC options. Default values will be used for options that are not specified.
        im_size: Size of the input image.
    Returns:
        samples: A list of returned HMC samples. Each sample is numpy float image of shape (im_size, im_size, 3).
    
    Note on image parametrization
    -----------------------------
    Images in lucid are represented as floating types lying in [0, 1].
    HMC, however, needs unbounded state spaces as states are updated by taking steps of size step_size. Hence we
    re-parametrize images Y in (0, 1)^N to X in R^N (N = width * heigth, the total dimensionality of the image), using
    the following one-to-one mapping:
        Y = sigmoid(X), X \in R^N, Y \in (0, 1)^N.
    Images are represented as Y, but the state space HMC will sample from is X.

    Probability is defined on Y. p(Y) is equal to the activation objective produced by Y. But when we perform HMC on X,
    we need to appropriately define p(X). We will need to multiply p(Y) by the determinant of the Jacobian of the
    transofmation from X to Y.
        p(X) dX = p(Y) dY
        => p(X) = p(Y) |dY/dX| = p(Y) * Jacobian_X(Y)
    Now,
        Jacobian_X(Y) = \prod_i d(Y_i)/d(X_i)
                      = \prod_i d(sigmoid(X_i))/d(X_i)
                      = \prod_i sigmoid(X_i) * \prod_i (1 - sigmoid(X_i))
    
    Then, \log Jacobian_X(Y) = 2 * \sum_i \log sigmoid(X_i) - \sum_i X_i

    \log p(X) = \log p(Y) + 2 * \sum_i \log sigmoid(X_i) - \sum_i X_i
    """
    # First step: optimization using optvis to find a good starting state.
    param_f = lambda: param.image(im_size)
    Y_start = optvis_render.render_vis(model, obj, param_f=param_f, verbose=False)[0]

    # Get X_start from Y_start by taking the inverse of the sigmoid.
    X_start = inverse_sigmoid(Y_start)

    def target_log_prob_fn(X_image):
        Y_image = tf.sigmoid(X_image)
        log_jacobian = tf.reduce_sum(2. * tf.log(Y_image) - X_image)

        Y_image, name_scope = import_graph(model, Y_image)
        
        def T(layer):
            if layer == "input": return Y_image
            if layer == "labels": return model.labels
            return Y_image.graph.get_tensor_by_name("{}/import/{}:0".format(name_scope, layer))

        log_p_Y = tf.log(obj(T))
        log_p_X = log_p_Y + log_jacobian

        return log_p_X

    # Run HMC.
    samples = hmc(X_start, target_log_prob_fn, hmc_opts)

    # Transform results from X_image to Y_image, via sigmoid.
    samples = [sigmoid(sample)[0] for sample in samples]
    return samples


def import_graph(model, t_input):
    default_graph = tf.get_default_graph()
    name_scope = default_graph.get_name_scope()
    import_scope = name_scope + '/import'
    assert default_graph.unique_name('import', False) == import_scope, '{}/import already exists!'.format(import_scope)

    t_input, t_prep_input = model.create_input(t_input, forget_xy_shape=True)
    tf.import_graph_def(
        model.graph_def, {model.input_name: t_prep_input}, name='import')

    model.post_import('import')

    return t_input, name_scope
