import numpy as np

from .linear_programming import lp_general_graph


def get_installed(method_filter=None):
    if method_filter is None:
        method_filter = ['ad3', 'qpbo', 'dai', 'ogm', 'lp']

    installed = []
    unary = np.zeros((1, 1))
    pw = np.zeros((1, 1))
    edges = np.empty((0, 2), dtype=np.int)
    for method in method_filter:
        try:
            inference_dispatch(unary, pw, edges, inference_method=method)
            installed.append(method)
        except ImportError:
            pass
    return installed


def compute_energy(unary_potentials, pairwise_potentials, edges, labels):
    """Compute energy of labels for given energy function.

    Convenience function with same interface as inference functions to easily
    compare solutions.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    labels : nd-array
        Variable assignment to evaluate.

    Returns
    -------
    energy : float
        Energy of assignment.
    """

    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    energy = np.sum(unary_potentials[np.arange(len(labels)), labels])
    for edge, pw in zip(edges, pairwise_potentials):
        energy += pw[labels[edge[0]], labels[edge[1]]]
    return energy


def inference_dispatch(unary_potentials, pairwise_potentials, edges,
                       inference_method, return_energy=False, **kwargs):
    """Wrapper function to dispatch between inference method by string.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    inference_method : string
        Possible choices currently are:
            * 'qpbo' for QPBO alpha-expansion (fast but approximate).
            * 'dai' for libDAI wrappers (default to junction tree).
            * 'lp' for build-in lp relaxation via GLPK (slow).
            * 'ad3' for AD^3 subgradient based dual solution of LP.
            * 'ogm' for OpenGM wrappers.
            * 'unary' for using unary potentials only.

        It is also possible to pass a tuple (string, dict) where the dict
        contains additional keyword arguments.

    relaxed : bool (default=False)
        Whether to return a relaxed solution (when appropriate)
        or round to the nearest integer solution. Only used for 'lp' and 'ad3'
        inference methods.

    return_energy : bool (default=False)
        Additionally return the energy of the returned solution (according to
        the solver).  If relaxed=False, this is the energy of the relaxed, not
        the rounded solution.

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
        If relaxed=True, this is a tuple of unary and pairwise "marginals"
        from the LP relaxation.
    """
    if isinstance(inference_method, tuple):
        additional_kwargs = inference_method[1]
        inference_method = inference_method[0]
        # append additional_kwargs, but take care not to modify the dicts we
        # got
        kwargs = dict(list(kwargs.items()) + list(additional_kwargs.items()))
    if inference_method == "qpbo":
        return inference_qpbo(unary_potentials, pairwise_potentials, edges,
                              **kwargs)
    elif inference_method == "gco":
        return inference_gco(unary_potentials, pairwise_potentials, edges,
                             **kwargs)
    elif inference_method == "dai":
        return inference_dai(unary_potentials, pairwise_potentials, edges,
                             return_energy=return_energy, **kwargs)
    elif inference_method == "lp":
        return inference_lp(unary_potentials, pairwise_potentials, edges,
                            return_energy=return_energy, **kwargs)
    elif inference_method == "ad3":
        return inference_ad3(unary_potentials, pairwise_potentials, edges,
                             return_energy=return_energy, **kwargs)
    elif inference_method == "ogm":
        return inference_ogm(unary_potentials, pairwise_potentials, edges,
                             return_energy=return_energy, **kwargs)
    elif inference_method == "unary":
        return inference_unaries(unary_potentials, pairwise_potentials, edges,
                                 **kwargs)
    else:
        raise ValueError("inference_method must be 'lp', 'ad3', 'qpbo', 'ogm'"
                         " or 'dai', got %s" % inference_method)


def _validate_params(unary_potentials, pairwise_params, edges):
    n_states = unary_potentials.shape[-1]
    if pairwise_params.shape == (n_states, n_states):
        # only one matrix given
        pairwise_potentials = np.repeat(pairwise_params[np.newaxis, :, :],
                                        edges.shape[0], axis=0)
    else:
        if pairwise_params.shape != (edges.shape[0], n_states, n_states):
            raise ValueError("Expected pairwise_params either to "
                             "be of shape n_states x n_states "
                             "or n_edges x n_states x n_states, but"
                             " got shape %s. n_states=%d, n_edge=%d."
                             % (repr(pairwise_params.shape), n_states,
                                edges.shape[0]))
        pairwise_potentials = pairwise_params
    return n_states, pairwise_potentials


def inference_ogm(unary_potentials, pairwise_potentials, edges,
                  return_energy=False, alg='dd', init=None,
                  reserveNumFactorsPerVariable=2, **kwargs):
    """Inference with OpenGM backend.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    alg : string
        Possible choices currently are:
            * 'bp' for Loopy Belief Propagation.
            * 'dd' for Dual Decomposition via Subgradients.
            * 'trws' for Vladimirs TRWs implementation.
            * 'trw' for OGM  TRW.
            * 'gibbs' for Gibbs sampling.
            * 'lf' for Lazy Flipper
            * 'fm' for Fusion Moves (alpha-expansion fusion)
            * 'dyn' for Dynamic Programming (message passing in trees)
            * 'gc' for Graph Cut
            * 'alphaexp' for Alpha Expansion using Graph Cuts
            * 'mqpbo' for multi-label qpbo

    init : nd-array
        Initial solution for starting inference (ignored by some algorithms).

    reserveNumFactorsPerVariable :
        reserve a certain number of factors for each variable can speed up
        the building of a graphical model.
        ( For a 2d grid with second order factors one should set this to 5
         4 2-factors and 1 unary factor for most pixels )

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
    """

    import opengm
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)
    n_nodes = len(unary_potentials)

    gm = opengm.gm(np.ones(n_nodes, dtype=opengm.label_type)*n_states)

    nFactors = int(n_nodes+edges.shape[0])
    gm.reserveFactors(nFactors)
    gm.reserveFunctions(nFactors,'explicit')
    #gm.reserveFactorsVariableIndices(n_nodes*n_states + int(edges.shape[0])*n_states**2 )

    # all unaries as one numpy array 
    # (opengm's value_type == float64 but all types are accepted)
    unaries = np.require(unary_potentials,dtype=opengm.value_type)*-1.0
    # add all unart functions at once
    fidUnaries = gm.addFunctions(unaries)
    visUnaries = np.arange(n_nodes,dtype=opengm.label_type)
    # add all unary factors at once
    gm.addFactors(fidUnaries,visUnaries)

    # add all pariwise functions at once 
    # - first axis of secondOrderFunctions iterates over the function)

    secondOrderFunctions = np.require(pairwise_potentials,dtype=opengm.value_type)*-1.0
    fidSecondOrder = gm.addFunctions(secondOrderFunctions)
    assert secondOrderFunctions.ndim == 3
    assert secondOrderFunctions.shape[1] == n_states
    assert secondOrderFunctions.shape[2] == n_states
    assert edges.ndim==2
    assert edges.shape[0]==secondOrderFunctions.shape[0]
    assert edges.shape[1]==2
    gm.addFactors(fidSecondOrder,edges.astype(np.uint64))


    #for i, un in enumerate(unary_potentials):
        #gm.addFactor(gm.addFunction(-un.astype(np.float32)), i)
    #for pw, edge in zip(pairwise_potentials, edges):
        #gm.addFactor(gm.addFunction(-pw.astype(np.float32)),
                     #edge.astype(np.uint64))


    if alg == 'bp':
        inference = opengm.inference.BeliefPropagation(gm)
    elif alg == 'dd':
        inference = opengm.inference.DualDecompositionSubgradient(gm)
    elif alg == 'trws':
        inference = opengm.inference.TrwsExternal(gm)
    elif alg == 'trw':
        inference = opengm.inference.TreeReweightedBp(gm)
    elif alg == 'gibbs':
        inference = opengm.inference.Gibbs(gm)
    elif alg == 'lf':
        inference = opengm.inference.LazyFlipper(gm)
    elif alg == 'icm':
        inference = opengm.inference.Icm(gm)
    elif alg == 'dyn':
        inference = opengm.inference.DynamicProgramming(gm)
    elif alg == 'fm':
        inference = opengm.inference.AlphaExpansionFusion(gm)
    elif alg == 'gc':
        inference = opengm.inference.GraphCut(gm)
    elif alg == 'loc':
        inference = opengm.inference.Loc(gm)
    elif alg == 'mqpbo':
        inference = opengm.inference.Mqpbo(gm)
    elif alg == 'alphaexp':
        inference = opengm.inference.AlphaExpansion(gm)
    if init is not None:
        inference.setStartingPoint(init)

    inference.infer()
    # we convert the result to int from unsigned int
    # because otherwise we are sure to shoot ourself in the foot
    res = inference.arg().astype(np.int)
    if return_energy:
        return res, gm.evaluate(res)  # inference.value() should also do the trick
    return res


def inference_gco(unary_potentials, pairwise_potentials, edges, **kwargs):
    from pygco import cut_from_graph_gen_potts
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    edges = edges.copy().astype(np.int32)
    pairwise_potentials = (1000 * pairwise_potentials).copy().astype(np.int32)

    pairwise_cost = {}
    for i in range(0, pairwise_potentials.shape[0]):
        cost = pairwise_potentials[i, 0, 0]
        if cost >= 0:
            pairwise_cost[(edges[i, 0], edges[i, 1])] = cost

    unary_potentials = (-1000 * unary_potentials).copy().astype(np.int32)

    if 'n_iter' in kwargs:
        y = cut_from_graph_gen_potts(unary_potentials, pairwise_cost, n_iter=kwargs['n_iter'])
    else:
        y = cut_from_graph_gen_potts(unary_potentials, pairwise_cost)

    return y[0].reshape(shape_org)


def inference_qpbo(unary_potentials, pairwise_potentials, edges, **kwargs):
    """Inference with PyQPBO backend.

    Used QPBO-I based move-making for undergenerating inference.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
    """

    from pyqpbo import alpha_expansion_general_graph
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unary_potentials = (-1000 * unary_potentials).copy().astype(np.int32)
    unary_potentials = unary_potentials.reshape(-1, n_states)
    pairwise_potentials = (-1000 * pairwise_potentials).copy().astype(np.int32)
    edges = edges.astype(np.int32).copy()
    y = alpha_expansion_general_graph(edges, unary_potentials,
                                      pairwise_potentials, random_seed=1)
    return y.reshape(shape_org)


def inference_dai(unary_potentials, pairwise_potentials, edges,
                  return_energy=False, alg='jt', **kwargs):
    """Inference with LibDAI backend.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    alg : string, (default='jt')
        Inference algorithm to use.
        Defaults to Junction Tree. THIS WILL BLOW UP for loopy graphs.

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
    """
    from daimrf import mrf
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    n_states = unary_potentials.shape[-1]
    log_unaries = unary_potentials.reshape(-1, n_states)
    max_entry = max(np.max(log_unaries), 1)
    unaries = np.exp(log_unaries / max_entry)

    y = mrf(unaries, edges.astype(np.int64),
            np.exp(pairwise_potentials / max_entry), alg=alg)
    y = y.reshape(shape_org)
    if return_energy:
        return y, compute_energy(unary_potentials, pairwise_potentials, edges,
                                 y)
    return y


def inference_lp(unary_potentials, pairwise_potentials, edges, relaxed=False,
                 return_energy=False, **kwargs):
    """Inference with build-in LP solver using GLPK backend.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    relaxed : bool (default=False)
        Whether to return the relaxed solution (``True``) or round to the next
        integer solution (``False``).

    return_energy : bool (default=False)
        Additionally return the energy of the returned solution (according to
        the solver).  If relaxed=False, this is the energy of the relaxed, not
        the rounded solution.

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
        If relaxed=False, this is a tuple of unary and edge 'marginals'.
    """
    shape_org = unary_potentials.shape[:-1]
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unaries = unary_potentials.reshape(-1, n_states)
    res = lp_general_graph(-unaries, edges, -pairwise_potentials)
    unary_marginals, pairwise_marginals, energy = res
    #n_fractional = np.sum(unary_marginals.max(axis=-1) < .99)
    #if n_fractional:
        #print("fractional solutions found: %d" % n_fractional)
    if relaxed:
        unary_marginals = unary_marginals.reshape(unary_potentials.shape)
        y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
        y = y.reshape(shape_org)
    if return_energy:
        return y, energy
    return y


def inference_ad3(unary_potentials, pairwise_potentials, edges, relaxed=False,
                  verbose=0, return_energy=False, branch_and_bound=False):
    """Inference with AD3 dual decomposition subgradient solver.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    relaxed : bool (default=False)
        Whether to return the relaxed solution (``True``) or round to the next
        integer solution (``False``).

    verbose : int (default=0)
        Degree of verbosity for solver.

    return_energy : bool (default=False)
        Additionally return the energy of the returned solution (according to
        the solver).  If relaxed=False, this is the energy of the relaxed, not
        the rounded solution.

    branch_and_bound : bool (default=False)
        Whether to attempt to produce an integral solution using
        branch-and-bound.

    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
        If relaxed=False, this is a tuple of unary and edge 'marginals'.
    """
    import ad3
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unaries = unary_potentials.reshape(-1, n_states)
    res = ad3.general_graph(unaries, edges, pairwise_potentials, verbose=1,
                            n_iterations=4000, exact=branch_and_bound)
    unary_marginals, pairwise_marginals, energy, solver_status = res
    if verbose:
        print(solver_status[0], end=' ')

    if solver_status in ["fractional", "unsolved"] and relaxed:
        unary_marginals = unary_marginals.reshape(unary_potentials.shape)
        y = (unary_marginals, pairwise_marginals)
    else:
        y = np.argmax(unary_marginals, axis=-1)
    if return_energy:
        return y, -energy
    return y


def inference_unaries(unary_potentials, pairwise_potentials, edges, verbose=0,
                      **kwargs):
    """Inference that only uses unary potentials.

    This methods can be used as a sanity check, as acceleration if no
    edges are present in an instance, and for debugging.

    Parameters
    ----------
    unary_potentials : nd-array
        Unary potentials of energy function.

    pairwise_potentials : nd-array
        Pairwise potentials of energy function.

    edges : nd-array
        Edges of energy function.

    verbose : int (default=0)
        Degree of verbosity for solver.


    Returns
    -------
    labels : nd-array
        Approximate (usually) MAP variable assignment.
        If relaxed=False, this is a tuple of unary and edge 'marginals'.
    """
    n_states, pairwise_potentials = \
        _validate_params(unary_potentials, pairwise_potentials, edges)

    unaries = unary_potentials.reshape(-1, n_states)
    return np.argmax(unaries, axis=-1)
