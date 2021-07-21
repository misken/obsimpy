__author__ = 'misken'

import numpy as np
import scipy.stats as stats
import scipy.optimize
import math


def poissoninv(prob, mean):
    """
    Return the cumulative inverse of the Poisson distribution.

    Useful for capacity planning approximations. Uses normal
    approximation to the Poisson distribution for mean > 50.

    Parameters
    ----------
    mean : float
        mean of the Poisson distribution
    prob :
        percentile desired

    Returns
    -------
    int
        minimum value, c,  such that P(X>c) <= prob

    """

    return stats.poisson.ppf(prob, mean)


def erlangb_direct(load, c):
    """
    Return the the probability of loss in M/G/c/c system.

    Parameters
    ----------
    load : float
        average arrival rate * average service time (units are erlangs)
    c : int
        number of servers


    Returns
    -------
    float
        probability arrival finds system full

    """

    p = stats.poisson.pmf(c, load) / stats.poisson.cdf(c, load)

    return p


def erlangb(load, c):
    """
    Return the the probability of loss in M/G/c/c system using recursive approach.

    Much faster than direct computation via
    scipy.stats.poisson.pmf(c, load) / scipy.stats.poisson.cdf(c, load)

    Parameters
    ----------
    load : float
        average arrival rate * average service time (units are erlangs)
    c : int
        number of servers

    Returns
    -------
    float
        probability arrival finds system full

    """

    invb = 1.0
    for j in range(1, c + 1):
        invb = 1.0 + invb * j / load

    b = 1.0 / invb

    return b


def erlangc(load, c):
    """
    Return the the probability of delay in M/M/c/inf system using recursive Erlang B approach.


    Parameters
    ----------
    load : float
        average arrival rate * average service time (units are erlangs)
    c : int
        number of servers

    Returns
    -------
    float
        probability all servers busy

    """

    rho = load / float(c)
    # if rho >= 1.0:
    #     raise ValueError("rho must be less than 1.0")
    
    eb = erlangb(load, c)
    ec = 1.0 / (rho + (1 - rho) * (1.0 / eb))

    return ec


def erlangcinv(prob, load):
    """
    Return the number of servers such that probability of delay in M/M/c/inf system is
    less than specified probability


    Parameters
    ----------
    prob : float
        threshold delay probability
    load : float
        average arrival rate * average service time (units are erlangs)

    Returns
    -------
    c : int
        number of servers

    """

    c = np.ceil(load)
    ec = erlangc(load, c)
    if ec <= prob:
        return c
    else:
        while ec > prob:
            c += 1
            ec = erlangc(load, c)

    return c


def mmc_prob_n(n, arr_rate, svc_rate, c):
    """
    Return the the probability of n customers in system in M/M/c/inf queue.

    Uses recursive approach from Tijms, H.C. (1994), "Stochastic Models: An Algorithmic Approach",
    John Wiley and Sons, Chichester (Section 4.5.1, p287)


    Parameters
    ----------
    n : int
        number of customers for which probability is desired
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        probability n customers in system (in service plus in queue)

    """

    rho = arr_rate / (svc_rate * float(c))

    # Step 0: Initialization - p[0] is initialized to one via creation method

    pbar = np.ones(max(n + 1, c))

    # Step 1: compute pbar

    for j in range(1, c):
        pbar[j] = arr_rate * pbar[j - 1] / (j * svc_rate)

    # Step 2: compute normalizing constant and normalize pbar

    gamma = np.sum(pbar) + rho * pbar[c - 1] / (1 - rho)
    p = pbar / gamma

    # Step 3: compute probs beyond c - 1

    for j in range(c, n + 1):
        p[j] = p[c - 1] * (rho ** (j - c + 1))

    return p[n]


def mmc_mean_qsize(arr_rate, svc_rate, c):
    """
    Return the the mean queue size in M/M/c/inf queue.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean number of customers in queue

    """

    rho = arr_rate / (svc_rate * float(c))

    mean_qsize = (rho ** 2 / (1 - rho) ** 2) * mmc_prob_n(c - 1, arr_rate, svc_rate, c)

    return mean_qsize


def mmc_mean_syssize(arr_rate, svc_rate, c):
    """
    Return the the mean system size in M/M/c/inf queue.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean number of customers in queue + service

    """

    load = arr_rate / svc_rate
    rho = load / float(c)

    mean_qsize = (rho ** 2 / (1 - rho) ** 2) * mmc_prob_n(c - 1, arr_rate, svc_rate, c)

    mean_syssize = mean_qsize + load

    return mean_syssize


def mmc_mean_qwait(arr_rate, svc_rate, c):
    """
    Return the the mean wait in queue time in M/M/c/inf queue.

    Uses mmc_mean_qsize along with Little's Law.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean wait time in queue

    """

    return mmc_mean_qsize(arr_rate, svc_rate, c) / arr_rate


def mmc_mean_systime(arr_rate, svc_rate, c):
    """
    Return the mean time in system (wait in queue + service time) in M/M/c/inf queue.

    Uses mmc_mean_qsize along with Little's Law (via mmc_mean_qwait) and relationship between W and Wq..

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean wait time in queue

    """

    return mmc_mean_qwait(arr_rate, svc_rate, c) + 1 / svc_rate


def mmc_prob_wait_normal(arr_rate, svc_rate, c):
    """
    Return the approximate probability of waiting (i.e. erlang C) in M/M/c/inf queue using a normal approximation.

    Uses normal approximation approach by Kolesar and Green, "Insights
    on Service System Design from a Normal Approximation to Erlang's
    Delay Formula", POM, V7, No3, Fall 1998, pp282-293

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        approximate probability of delay in queue

    """

    load = arr_rate / svc_rate

    prob_wait = 1.0 - stats.norm.cdf(c - load - 0.5) / np.sqrt(load)

    return prob_wait


def mgc_prob_wait_erlangc(arr_rate, svc_rate, c):
    """
    Return the approximate probability of waiting in M/G/c/inf queue using Erlang-C as approximation.

    It's well known that the Erlang-C formula, P(W>0) in M/M/c is a good approximation for
    P(W>0) in M/G/c. See, for example, Tjims (1994) on p296 or Whitt (1993) "Approximations
    for the GI/G/m queue", Production and Operations Management, 2, 2.


    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        approximate probability of delay in queue

    """

    load = arr_rate / svc_rate

    prob_wait = erlangc(load, c)

    return prob_wait


def mm1_qwait_cdf(t, arr_rate, svc_rate):
    """
    Return P(Wq < t) in M/M/1/inf queue.


    Parameters
    ----------
    t : float
        wait time of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.


    Returns
    -------
    float
        probability wait time in queue is < t

    """

    rho = arr_rate / svc_rate

    term1 = rho
    term2 = -svc_rate * (1 - rho) * t

    prob_wq_lt_t = 1.0 - term1 * np.exp(term2)

    return prob_wq_lt_t


def mmc_qwait_cdf(t, arr_rate, svc_rate, c):
    """
    Return P(Wq < t) in M/M/c/inf queue.


    Parameters
    ----------
    t : float
        wait time of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers


    Returns
    -------
    float
        probability wait time in queue is < t

    """

    rho = arr_rate / (svc_rate * float(c))

    term1 = rho / (1 - rho)
    term2 = mmc_prob_n(c - 1, arr_rate, svc_rate, c)
    term3 = -c * svc_rate * (1 - rho) * t

    prob_wq_lt_t = 1.0 - term1 * term2 * np.exp(term3)

    return prob_wq_lt_t


def mmc_qwait_cdf_inv(t, prob, arr_rate, svc_rate):
    """
    Return the number of servers such that probability of delay < t in M/M/c/inf system is
    greater than specified prob


    Parameters
    ----------
    t : float
        wait time threshold
    prob : float
        threshold delay probability
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.

    Returns
    -------
    c : int
        number of servers

    """

    c = math.ceil(arr_rate / svc_rate)
    pwait_lt_t = mmc_qwait_cdf(t, arr_rate, svc_rate, c)
    if pwait_lt_t >= prob:
        return c
    else:
        while pwait_lt_t < prob:
            c += 1
            pwait_lt_t = mmc_qwait_cdf(t, arr_rate, svc_rate, c)

    return c


def mm1_qwait_pctile(p, arr_rate, svc_rate):
    """
    Return p'th percentile of P(Wq < t) in M/M/1/inf queue.


    Parameters
    ----------
    p : float
        percentile of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.


    Returns
    -------
    float
        t such that P(wait time in queue is < t) = p

    """

    # For initial guess, we'll use percentile from similar M/M/1 system
    init_guess = 1/svc_rate


    waitq_pctile = scipy.optimize.newton(_mm1_waitq_pctile_wrap,init_guess,args=(p, arr_rate, svc_rate))

    return waitq_pctile


def _mm1_waitq_pctile_wrap(t, p, arr_rate, svc_rate):
    return mm1_qwait_cdf(t, arr_rate, svc_rate) - p


def mmc_qwait_pctile(p, arr_rate, svc_rate, c):
    """
    Return p'th percentile of P(Wq < t) in M/M/c/inf queue.


    Parameters
    ----------
    p : float
        percentile of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers


    Returns
    -------
    float
        t such that P(wait time in queue is < t) = p

    """

    # For initial guess, we'll use percentile from similar M/M/1 system

    init_guess = mm1_qwait_pctile(p, arr_rate, c * svc_rate)

    waitq_pctile = scipy.optimize.newton(_mmc_waitq_pctile_wrap,init_guess,args=(p, arr_rate, svc_rate, c))

    return waitq_pctile


def _mmc_waitq_pctile_wrap(t, p, arr_rate, svc_rate, c):
    return mmc_qwait_cdf(t, arr_rate, svc_rate, c) - p


def mdc_mean_qwait_cosmetatos(arr_rate, svc_rate, c):
    """
    Return the approximate mean queue wait in M/D/c/inf queue using Cosmetatos approximation.

    See Cosmetatos, George P. "Approximate explicit formulae for the average queueing time in the processes (M/D/r)
    and (D/M/r)." Infor 13.3 (1975): 328-331.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean number of customers in queue

    """

    rho = arr_rate / (svc_rate * float(c))

    term1 = 0.5
    term2 = (c - 1) * (np.sqrt(4 + 5 * c) - 2) / (16 * c)
    term3 = (1 - rho) / rho
    term4 = mmc_mean_qwait(arr_rate, svc_rate, c)

    mean_qwait = term1 * (1 + term2 * term3) * term4

    return mean_qwait


def mdc_mean_qsize_cosmetatos(arr_rate, svc_rate, c):
    """
    Return the approximate mean queue size in M/D/c/inf queue using Cosmetatos approximation.

    See Cosmetatos, George P. "Approximate explicit formulae for the average queueing time in the processes (M/D/r)
    and (D/M/r)." Infor 13.3 (1975): 328-331.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers

    Returns
    -------
    float
        mean number of customers in queue

    """

    mean_qwait = mdc_mean_qwait_cosmetatos(arr_rate, svc_rate, c)
    mean_qsize = mean_qwait * arr_rate

    return mean_qsize


def mgc_mean_qwait_kimura(arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return the approximate mean queue wait in M/G/c/inf queue using Kimura approximation.

    See Kimura, Toshikazu. "Approximations for multi-server queues: system interpolations."
    Queueing Systems 17.3-4 (1994): 347-382.

    It's based on interpolation between an M/D/c and a M/M/c queueing system.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    term1 = 1.0 + cv2_svc_time
    term2 = 2.0 * cv2_svc_time / mmc_mean_qwait(arr_rate, svc_rate, c)
    term3 = (1.0 - cv2_svc_time) / mdc_mean_qwait_cosmetatos(arr_rate, svc_rate, c)

    mean_qwait = term1 / (term2 + term3)

    return mean_qwait


def mgc_mean_qsize_kimura(arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return the approximate mean queue size in M/G/c/inf queue using Kimura approximation.

    See Kimura, Toshikazu. "Approximations for multi-server queues: system interpolations."
    Queueing Systems 17.3-4 (1994): 347-382.

    It's based on interpolation between an M/D/c and a M/M/c queueing system.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean number of customers in queue

    """

    mean_qwait = mgc_mean_qwait_kimura(arr_rate, svc_rate, c, cv2_svc_time)
    mean_qsize = mean_qwait * arr_rate

    return mean_qsize


def mgc_qwait_cdf_whitt(t, arr_rate, svc_rate, c, cs2):
    """
    Return the approximate P(Wq <= t) in M/G/c/inf queue using Whitt's G/C/c approximation.

    Comparison of Whitt's approximation with the van Hoorn and Tijms M/G/c specific approximation suggests that using
    Whitt's is sufficiently accurate and much easier in that we don't have to numerically integrate
    excess service time distributions.

    Whitt, Ward. "Approximations for the GI/G/m queue" Production and Operations Management 2, 2
    (Spring 1993): 114-161.

    van Hoorn, Michiel Harpert, and Hendrik Cornelis Tijms. "Approximations for the waiting time
    distribution of the M/G/c queue." Performance Evaluation 2.1 (1982): 22-28.



    Parameters
    ----------
    t : float
        wait time of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        ~ P(Wq <= t)

    """


    pwait_lt_t = ggm_qwait_cdf_whitt(t, arr_rate, svc_rate, c, 1.0, cs2)

    return pwait_lt_t


def mgc_mean_qwait_bjorklund(arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return the approximate mean queue wait in M/G/c/inf queue using Bjorklund and Elldin approximation.

    See Kimura, Toshikazu. "Approximations for multi-server queues: system interpolations."
    Queueing Systems 17.3-4 (1994): 347-382.

    It's based on interpolation between an M/D/c and a M/M/c queueing system.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean number of customers in queue

    """

    term1 = cv2_svc_time * mmc_mean_qwait(arr_rate, svc_rate, c)
    term2 = (1.0 - cv2_svc_time) * mdc_mean_qwait_cosmetatos(arr_rate, svc_rate, c)

    mean_qwait = term1 + term2

    return mean_qwait


def mgc_mean_qsize_bjorklund(arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return the approximate mean queue size in M/G/c/inf queue using Bjorklund and Elldin approximation.

    See Kimura, Toshikazu. "Approximations for multi-server queues: system interpolations."
    Queueing Systems 17.3-4 (1994): 347-382.

    It's based on interpolation between an M/D/c and a M/M/c queueing system.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean number of customers in queue

    """

    mean_qwait = mgc_mean_qwait_bjorklund(arr_rate, svc_rate, c, cv2_svc_time)
    mean_qsize = mean_qwait * arr_rate

    return mean_qsize


def mgc_qcondwait_pctile_firstorder_2moment(prob, arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return an approximate conditional queue wait percentile in M/G/c/inf system.

    The approximation is based on a first order approximation using the M/M/c delay percentile.
    See Tijms, H.C. (1994), "Stochastic Models: An Algorithmic Approach", John Wiley and Sons, Chichester
    Chapter 4, p299-300

    The percentile is conditional on Wq>0 (i.e. on event customer waits)

    This 1st order approximation is OK for 0<=CVSquared<=2 and prob>1-Prob(Delay)
    Note that for Prob(Delay) we use MMC as approximation for same quantity in MGC.
    Justification in Tijms (p296)


    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        t such that P(wait time in queue is < t | wait time in queue is > 0) = prob

    """

    load = arr_rate / svc_rate
    # Compute corresponding prob for unconditional wait (see p274 of Tjims)
    equivalent_uncond_prob = 1.0 - (1.0 - prob) * erlangc(load, c)
    # Compute conditional wait time percentile for M/M/c system to use in approximation
    condwaitq_pctile_mmc = mmc_qwait_pctile(equivalent_uncond_prob, arr_rate, svc_rate, c)
    # First order approximation for conditional wait time in queue
    condwaitq_pctile = 0.5 * (1.0 + cv2_svc_time) * condwaitq_pctile_mmc

    return condwaitq_pctile


def mgc_qcondwait_pctile_secondorder_2moment(prob, arr_rate, svc_rate, c, cv2_svc_time):
    """
    Return an approximate conditional queue wait percentile in M/G/c/inf system.

    The approximation is based on a second order approximation using the M/M/c delay percentile.
    See Tijms, H.C. (1994), "Stochastic Models: An Algorithmic Approach", John Wiley and Sons, Chichester
    Chapter 4, p299-300

    The percentile is conditional on Wq>0 (i.e. on event customer waits)

    This approximation is based on interpolation between corresponding M/M/c and M/D/c systems.


    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        t such that P(wait time in queue is < t | wait time in queue is > 0) = prob

    """

    load = arr_rate / svc_rate
    # Compute corresponding prob for unconditional wait (see p274 of Tjims)
    equivalent_uncond_prob = 1.0 - (1.0 - prob) * erlangc(load, c)

    # Compute conditional wait time percentile for M/M/c system to use in approximation
    condwaitq_pctile_mmc = mmc_qwait_pctile(equivalent_uncond_prob, arr_rate, svc_rate, c)

    # Compute conditional wait time percentile for M/D/c system to use in approximation
    # TODO: implement mdc_qwait_pctile
    condqwait_pctile_mdc = mdc_waitq_pctile(equivalent_uncond_prob, arr_rate, svc_rate, c)

    # Second order approximation for conditional wait time in queue
    condwaitq_pctile = (1.0 - cv2_svc_time) * condqwait_pctile_mdc + cv2_svc_time * condwaitq_pctile_mmc

    return condwaitq_pctile


def mg1_mean_qsize(arr_rate, svc_rate, cv2_svc_time):
    """
    Return the mean queue size in M/G/1/inf queue using P-K formula.

    See any decent queueing book.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean number of customers in queue

    """

    rho = arr_rate / svc_rate
    mean_qsize = (arr_rate ** 2) * cv2_svc_time/(2 * (1.0 - rho))

    return mean_qsize


def mg1_mean_qwait(arr_rate, svc_rate, cs2):
    """
    Return the mean queue wait in M/G/1/inf queue using P-K formula along with Little's Law.

    See any decent queueing book.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue
    """

    mean_qsize = mg1_mean_qsize(arr_rate, svc_rate, cs2)
    mean_qwait = mean_qsize / arr_rate

    return mean_qwait


def gamma_0(m, rho):
    """
    See p124 immediately after Eq 2.16.

    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """

    term1 = 0.24
    term2 = (1 - rho) * (m - 1) * (math.sqrt(4 + 5 * m) - 2 ) / (16 * m * rho)

    return min(term1, term2)


def _ggm_mean_qwait_whitt_phi_1(m, rho):
    """
    See p124 immediately after Eq 2.16.

    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """

    return 1.0 + gamma_0(m, rho)


def _ggm_mean_qwait_whitt_phi_2(m, rho):
    """
    See p124 immediately after Eq 2.18.

    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """

    return 1.0 - 4.0 * gamma_0(m, rho)


def _ggm_mean_qwait_whitt_phi_3(m, rho):
    """
    See p124 immediately after Eq 2.20.

    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """

    term1 = _ggm_mean_qwait_whitt_phi_2(m, rho)
    term2 = math.exp(-2.0 * (1 - rho) / (3.0 * rho))

    return term1 * term2


def _ggm_mean_qwait_whitt_phi_4(m, rho):
    """
    See p125 , Eq 2.21.

    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """
    term1 = 1.0
    term2 = 0.5 * (_ggm_mean_qwait_whitt_phi_1(m, rho) + _ggm_mean_qwait_whitt_phi_3(m, rho))
    return min(term1, term2)


def _ggm_mean_qwait_whitt_psi_0(c2, m, rho):
    """
    See p125 , Eq 2.22.

    :param c2: float
        common squared CV for both arrival and service process
    :param m: int
        number of servers
    :param rho: float
        lambda / (mu * m)
    :return: float

    """

    if c2 >= 1:
        return 1.0
    else:
        return _ggm_mean_qwait_whitt_phi_4(m, rho) ** (2 * (1 - c2))


def _ggm_mean_qwait_whitt_phi_0(rho, ca2, cs2, m):
    """
    See p125 , Eq 2.25.

    :param rho: float
        lambda / (mu * m)
    :param ca2: float
        squared CV for arrival process
    :param cs2: float
        squared CV for service process
    :param m: int
        number of servers

    :return: float

    """

    if ca2 >= cs2:
        term1 = _ggm_mean_qwait_whitt_phi_1(m, rho) * (4 * (ca2 - cs2) / (4 * ca2 - 3 * cs2))
        term2 = (cs2 / (4 * ca2 - 3 * cs2)) * _ggm_mean_qwait_whitt_psi_0((ca2 + cs2) / 2.0, m, rho)
        return term1 + term2
    else:
        term1 = _ggm_mean_qwait_whitt_phi_3(m, rho) * ((cs2 - ca2) / (2 * ca2 + 2 * cs2))
        term2 = ( (cs2 + 3 * ca2) / (2 * ca2 + 2 * cs2) )
        term3 = _ggm_mean_qwait_whitt_psi_0((ca2 + cs2) / 2.0, m, rho)
        check = term2 * term3 / term1
        #print (check)
        return term1 + term2 * term3


def ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate mean queue wait in GI/G/c/inf queue using Whitt's 1993 approximation.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    It's based on interpolations with corrections between an M/D/c, D/M/c and a M/M/c queueing systems.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    rho = arr_rate / (svc_rate * float(m))
    if rho >= 1.0:
        raise ValueError("rho must be less than 1.0")

    # Now implement Eq 2.24 on p 125

    # Hack - for some reason I can't get this approximation to match Table 2 in the above
    # reference for the case of D/M/m. However, if I use Eq 2.20 (specific for the D/M/m case),
    # I do match the expected results. So, for now, I'll trap for this case.

    if ca2 == 0 and cs2 == 1:
        qwait = dmm_mean_qwait_whitt(arr_rate, svc_rate, m)

    else:
        term1 = _ggm_mean_qwait_whitt_phi_0(rho, ca2, cs2, m)
        term2 = 0.5 * (ca2 + cs2)
        term3 = mmc_mean_qwait(arr_rate, svc_rate, m)

        qwait = term1 * term2 * term3

    return qwait


def ggm_prob_wait_whitt(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate P(Wq > 0) in GI/G/c/inf queue using Whitt's 1993 approximation.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    It's based on interpolations with corrections between an M/D/c, D/M/c and a M/M/c queueing systems.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    rho = arr_rate / (svc_rate * float(m))

    # For ca2 = 1 (e.g. Poisson arrivals), Whitt uses fact that Erlang-C works well for M/G/c

    if ca2 == 1:
        pwait = mgc_prob_wait_erlangc(arr_rate, svc_rate, m)

    else:
        pi = _ggm_prob_wait_whitt_pi(m, rho, ca2, cs2)
        pwait = min(pi, 1)

    return pwait


def _ggm_prob_wait_whitt_z(ca2, cs2):
    """
    Equation 3.8 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.


    Parameters
    ----------
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        approximation for intermediate term z (see Eq 3.6)

    """

    z = (ca2 + cs2) / (1.0 + cs2)

    return z


def _ggm_prob_wait_whitt_gamma(m, rho, z):
    """
    Equation 3.5 on p136 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    z : float
        intermediate term approximated in Eq 3.8

    Returns
    -------
    float
        intermediate term gamma (see Eq 3.5)

    """

    term1 = m - m * rho - 0.5
    term2 = np.sqrt(m * rho * z)
    gamma = term1 / term2

    return gamma


def _ggm_prob_wait_whitt_pi_6(m, rho, z):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    z : float
        intermediate term approximated in Eq 3.8

    Returns
    -------
    float
        intermediate term pi_6 (see Eq 3.11)

    """

    pi_6 = 1.0 - stats.norm.cdf((m - m * rho - 0.5) / np.sqrt(m * rho * z))

    return pi_6


def _ggm_prob_wait_whitt_pi_5(m, rho, ca2, cs2):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_5(see Eq 3.11)

    """

    term1 = 2.0 * (1.0 - rho) * np.sqrt(m) / (1.0 + ca2)
    term2 = (1.0 - rho) * np.sqrt(m)

    term3 = erlangc(rho * m, m) * (1.0 - stats.norm.cdf(term1)) / (1.0 - stats.norm.cdf(term2))

    pi_5 = min(1.0,term3)

    return pi_5


def _ggm_prob_wait_whitt_pi_4(m, rho, ca2, cs2):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_5(see Eq 3.11)

    """

    term1 = (1.0 + cs2) * (1.0 - rho) * np.sqrt(m) / (ca2 + cs2)
    term2 = (1.0 - rho) * np.sqrt(m)

    term3 = erlangc(rho * m, m) * (1.0 - stats.norm.cdf(term1)) / (1.0 - stats.norm.cdf(term2))

    pi_4 = min(1.0,term3)

    return pi_4


def _ggm_prob_wait_whitt_pi_1(m, rho, ca2, cs2):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_5(see Eq 3.11)

    """

    pi_4 = _ggm_prob_wait_whitt_pi_4(m, rho, ca2, cs2)
    pi_5 = _ggm_prob_wait_whitt_pi_5(m, rho, ca2, cs2)

    pi_1 = (rho ** 2) * pi_4 + (1.0 - rho **2) * pi_5

    return pi_1


def _ggm_prob_wait_whitt_pi_2(m, rho, ca2, cs2):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_2(see Eq 3.11)

    """

    pi_1 = _ggm_prob_wait_whitt_pi_1(m, rho, ca2, cs2)
    z = _ggm_prob_wait_whitt_z(ca2, cs2)
    pi_6 = _ggm_prob_wait_whitt_pi_6(m, rho, z)

    pi_2 = ca2 * pi_1 + (1.0 - ca2) * pi_6

    return pi_2


def _ggm_prob_wait_whitt_pi_3(m, rho, ca2, cs2):
    """
    Part of Equation 3.11 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_5(see Eq 3.11)

    """
    z = _ggm_prob_wait_whitt_z(ca2, cs2)
    gamma = _ggm_prob_wait_whitt_gamma(m, rho, z)
    pi_2 = _ggm_prob_wait_whitt_pi_2(m, rho, ca2, cs2)
    pi_1 = _ggm_prob_wait_whitt_pi_1(m, rho, ca2, cs2)

    term1 = 2.0 * (1.0 - ca2) * (gamma - 0.5)
    term2 = 1.0 - term1

    pi_3 = term1 * pi_2 + term2 * pi_1

    return pi_3


def _ggm_prob_wait_whitt_pi(m, rho, ca2, cs2):
    """
    Equation 3.10 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        intermediate term pi_5(see Eq 3.11)

    """
    z = _ggm_prob_wait_whitt_z(ca2, cs2)
    gamma = _ggm_prob_wait_whitt_gamma(m, rho, z)

    if m <= 6 or gamma <= 0.5 or ca2 >= 1:
        pi = _ggm_prob_wait_whitt_pi_1(m, rho, ca2, cs2)
    elif m >= 7 and gamma >= 1.0 and ca2 < 1:
        pi = _ggm_prob_wait_whitt_pi_2(m, rho, ca2, cs2)
    else:
        pi = _ggm_prob_wait_whitt_pi_3(m, rho, ca2, cs2)

    return pi


def _ggm_prob_wait_whitt_whichpi(m, rho, ca2, cs2):
    """
    Equation 3.10 on p139 of Whitt (1993). Used in approximation for P(Wq > 0) in GI/G/c/inf queue.

    Primarily used for debugging and validation of the approximation implementation.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    m : int
        number of servers
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    int
        the pi case used in the approximation (1, 2, or 3)

    """
    z = _ggm_prob_wait_whitt_z(ca2, cs2)
    gamma = _ggm_prob_wait_whitt_gamma(m, rho, z)

    if m <= 6 or gamma <= 0.5 or ca2 >= 1:
        whichpi = 1
    elif m >= 7 and gamma >= 1.0 and ca2 < 1:
        whichpi = 2
    else:
        whichpi = 3

    return whichpi


def _ggm_qcondwait_whitt_ds3(cs2):
    """
    Return the approximate E(V^3)/(EV)^2 where V is a service time; based on either a hyperexponential
    or Erlang distribution. Used in approximation of conditional wait time CDF (conditional on W>0).

    Whitt refers to conditional wait as D in his paper:

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    This is Equation 4.3 on p146. Note that there is a typo in the original paper in which the first term
    for Case 1 is shown as cubed, whereas it should be squared. This can be confirmed by seeing Eq 51 in
    Whitt's paper on the QNA (Bell Systems Technical Journal, Nov 1983).

    Parameters
    ----------
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    if cs2 >= 1:
        ds3 = 3.0 * cs2 * (1.0 + cs2)
    else:
        ds3 = (2 * cs2 + 1.0) * (cs2 + 1.0)


    return ds3


def ggm_qcondwait_whitt_cd2(rho, cs2):
    """
    Return the approximate squared coefficient of conditional wait time (aka delay) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    This is Equation 4.2 on p145.

    Parameters
    ----------
    rho : float
        traffic intensity; arr_rate / (svc_rate * m)

    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    term1 = 2 * rho - 1.0
    term2 = 4 * (1.0 - rho) * _ggm_qcondwait_whitt_ds3(cs2)
    term3 = 3.0 * (cs2 + 1.0) ** 2

    cd2 = term1+ term2 / term3

    return cd2


def ggm_qwait_whitt_cw2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate squared coefficient of wait time in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        scv of wait time in queue

    """

    rho = arr_rate / (svc_rate * float(m))
    pwait = ggm_prob_wait_whitt(arr_rate, svc_rate, m, ca2, cs2)
    cd2 = ggm_qcondwait_whitt_cd2(rho, cs2)

    cw2 = (cd2 + 1 - pwait) / pwait

    return cw2


def ggm_qcondwait_whitt_ed(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate mean conditional wait time (aka delay) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    pwait = ggm_prob_wait_whitt(arr_rate, svc_rate, m, ca2, cs2)
    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2) / pwait

    return meanwait


def ggm_qcondwait_whitt_vard(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate variance of conditional wait time (aka delay) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    rho = arr_rate / (svc_rate * float(m))
    pwait = ggm_prob_wait_whitt(arr_rate, svc_rate, m, ca2, cs2)
    cd2 = ggm_qcondwait_whitt_cd2(rho, cs2)
    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    vard = (meanwait ** 2) * cd2 / (pwait ** 2)

    return vard


def ggm_qcondwait_whitt_ed2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate 2nd moment of conditional wait time (aka delay) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    pwait = ggm_prob_wait_whitt(arr_rate, svc_rate, m, ca2, cs2)
    vard = ggm_qcondwait_whitt_vard(arr_rate, svc_rate, m, ca2, cs2)
    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    # Compute conditional wait
    meandelay = meanwait / pwait

    ed2 = vard + meandelay ** 2

    return ed2


def ggm_qwait_whitt_varw(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate variance of wait time (aka delay) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    cw2 = ggm_qwait_whitt_cw2(arr_rate, svc_rate, m, ca2, cs2)
    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    varw = (meanwait ** 2) * cw2

    return varw


def ggm_qwait_whitt_ew2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate 2nd moment of wait time in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    varw = ggm_qwait_whitt_varw(arr_rate, svc_rate, m, ca2, cs2)
    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    ew2 = varw + meanwait ** 2

    return ew2


def ggm_mean_sojourn_whitt(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate soujourn time (wait + service) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    meanwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    sojourn = meanwait + 1.0 / svc_rate

    return sojourn


def ggm_sojourn_whitt_var(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate variance of soujourn time (wait + service) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    varwait = ggm_qwait_whitt_varw(arr_rate, svc_rate, m, ca2, cs2)

    sojourn = varwait + cs2 * (1.0 / svc_rate) ** 2

    return sojourn


def ggm_sojourn_whitt_et2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate 2nd moment of soujourn time (wait + service) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    varsojourn = ggm_sojourn_whitt_var(arr_rate, svc_rate, m, ca2, cs2)
    meansojourn = ggm_mean_sojourn_whitt(arr_rate, svc_rate, m, ca2, cs2)

    et2 = varsojourn + meansojourn ** 2

    return et2


def ggm_sojourn_whitt_cv2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate scv of soujourn time (wait + service) in G/G/m queue

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        variance of conditional wait time in queue

    """

    varsojourn = ggm_sojourn_whitt_var(arr_rate, svc_rate, m, ca2, cs2)
    meansojourn = ggm_mean_sojourn_whitt(arr_rate, svc_rate, m, ca2, cs2)

    cv2 = varsojourn / meansojourn ** 2

    return cv2


def ggm_mean_qsize_whitt(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate mean queue size in GI/G/c/inf queue using Whitt's 1993 approximation and Little's Law.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    It's based on interpolations with corrections between an M/D/c, D/M/c and a M/M/c queueing systems.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    # Use Eq 2.24 on p 125 to compute mean wait time in queue
    qwait = ggm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2, cs2)

    # Now use Little's Law
    return qwait * arr_rate


def ggm_mean_syssize_whitt(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate mean system size in GI/G/c/inf queue using Whitt's 1993 approximation and Little's Law.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    It's based on interpolations with corrections between an M/D/c, D/M/c and a M/M/c queueing systems.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        mean wait time in queue

    """

    # Use Eq 2.24 on p 125 to compute mean wait time in queue
    mean_sojourn = ggm_mean_sojourn_whitt(arr_rate, svc_rate, m, ca2, cs2)

    # Now use Little's Law
    return mean_sojourn * arr_rate


def dmm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2=0.0, cs2=1.0):
    """
    Return the approximate mean queue size in D/M/m/inf queue using Whitt's 1993 approximation.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161. Specifically, this approximation
    is Eq 2.20 on p124.

    This, along with mdm_mean_qwait_whitt are refinements of the Cosmetatos approximations.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution (0 for D)
    cs2 : float
        squared coefficient of variation for service time distribution (1 for M)

    Returns
    -------
    float
        mean wait time in queue

    """



    rho = arr_rate / (svc_rate * float(m))

    # Now implement Eq 2.20 on p 124

    term1 = _ggm_mean_qwait_whitt_phi_3(m, rho)
    term2 = 0.5 * (ca2 + cs2)
    term3 = mmc_mean_qwait(arr_rate, svc_rate, m)

    return term1 * term2 * term3


def mdm_mean_qwait_whitt(arr_rate, svc_rate, m, ca2=0.0, cs2=1.0):
    """
    Return the approximate mean queue size in M/D/m/inf queue using Whitt's 1993 approximation.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161. Specifically, this approximation
    is Eq 2.16 on p124.

    This, along with dmm_mean_qwait_whitt are refinements of the Cosmetatos approximations.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution (0 for D)
    cs2 : float
        squared coefficient of variation for service time distribution (1 for M)

    Returns
    -------
    float
        mean wait time in queue

    """


    rho = arr_rate / (svc_rate * float(m))

    # Now implement Eq 2.16 on p 124

    term1 = _ggm_mean_qwait_whitt_phi_1(m, rho)
    term2 = 0.5 * (ca2 + cs2)
    term3 = mmc_mean_qwait(arr_rate, svc_rate, m)

    return term1 * term2 * term3


def fit_balanced_hyperexpon2(mean, cs2):
    """
    Return the branching probability and rates for a balanced H2 distribution based
    on a specified mean and scv. Intended for scv's > 1.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    Parameters
    ----------
    cs2 : float
        squared coefficient of variation for desired distribution

    Returns
    -------
    tuple (float p, float rate1, float rate2)
        branching probability and exponential rates

    """

    p1 = 0.5 * (1 + np.sqrt((cs2-1) / (cs2+1)))
    p2 = 1 - p1
    mu1 = 2 * p1 / mean
    mu2 = 2 * p2 / mean

    return (p1, mu1, mu2)


def hyperexpon_cdf(x, probs, rates):
    """
    Return the P(X < x) where X is hypergeometric with probabilities and exponential rates
    in lists probs and rates.

    Parameters
    ----------
    probs : list of floats
        branching probabilities for hyperexponential

    probs : list of floats
        exponential rates

    Returns
    -------
    float
        P(X<x) where X~hyperexponetial(probs, rates)

    """

    sumproduct = sum([p * np.exp(-r * x) for (p, r) in zip(probs, rates)])
    prob_lt_x = 1.0 - sumproduct

    return prob_lt_x


def ggm_qcondwait_cdf_whitt(t, arr_rate, svc_rate, c, ca2, cs2):
    """
    Return the approximate P(D <= t) where D = (W|W>0) in G/G/m queue using Whitt's two moment
    approximation.

    See Section 4 of Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    It's based on an approach he originally used for G/G/1 queues in QNA. There are different
    cases based on the value of an approximation for the scv of D.

    Parameters
    ----------
    t : float
        wait time of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        ~ P(D <= t | )

    """

    rho = arr_rate / (svc_rate * float(c))

    ed = ggm_mean_qwait_whitt(arr_rate, svc_rate, c, ca2, cs2) / ggm_prob_wait_whitt(arr_rate, svc_rate, c, ca2, cs2)

    cd2 = ggm_qcondwait_whitt_cd2(rho,cs2)

    if cd2 > 1.01:
        # Hyperexponential approx

        p1, gamma1, gamma2 = fit_balanced_hyperexpon2(ed, cd2)
        p2 = 1.0 - p1

        prob_wait_ltx = hyperexpon_cdf(t, [p1,p2], [gamma1, gamma2])

    elif cd2 >= 0.99 and cd2 <= 1.01:
        # Exponential approx
        prob_wait_ltx = stats.expon.cdf(t,scale=ed)

    elif cd2 >= 0.501 and cd2 < 0.99:
        # Convolution of two exponentials approx
        vard = ggm_qcondwait_whitt_vard(arr_rate, svc_rate, c, ca2, cs2)
        gamma2 = 2.0 / (ed + np.sqrt(2 * vard - ed ** 2))
        gamma1 = 1.0 / (ed - 1.0 / gamma2)

        prob_wait_gtx = (gamma1 * np.exp(-gamma2 * t) - gamma2 * np.exp(-gamma1 * t)) / (gamma1 - gamma2)
        prob_wait_ltx = 1.0 - prob_wait_gtx
    else:
        # Erlang approx
        gamma1 = 2.0 / ed

        prob_wait_gtx = np.exp(-gamma1 * t) * (1.0 + gamma1 * t)
        prob_wait_ltx = 1.0 - prob_wait_gtx

    return prob_wait_ltx


def ggm_qwait_cdf_whitt(t, arr_rate, svc_rate, c, ca2, cs2):
    """
    Return the approximate P(W <= t) in G/G/m queue using Whitt's two moment
    approximation for conditional wait and the P(W>0).

    See Section 4 of Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.

    See ggm_qcondwait_cdf_whitt for more details.

    Parameters
    ----------
    t : float
        wait time of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        ~ P(W <= t | )

    """


    qcondwait = ggm_qcondwait_cdf_whitt(t, arr_rate, svc_rate, c, ca2, cs2)
    pdelay = ggm_prob_wait_whitt(arr_rate, svc_rate, c, ca2, cs2)

    qwait = qcondwait * pdelay + (1.0 - pdelay)

    return qwait


def ggm_qwait_pctile_whitt(p, arr_rate, svc_rate, c, ca2, cs2):
    """
    Return approx p'th percentile of P(Wq < t) in G/G/c/inf queue using Whitt's two moment
    approximation for the wait time CDF


    Parameters
    ----------
    p : float
        percentile of interest
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers


    Returns
    -------
    float
        t such that P(wait time in queue is < t) = p

    """

    # For initial guess, we'll use percentile from similar M/M/1 system

    init_guess = mm1_qwait_pctile(p, arr_rate, c * svc_rate)

    waitq_pctile = scipy.optimize.newton(_ggm_waitq_pctile_whitt_wrap,init_guess,args=(p, arr_rate, svc_rate, c, ca2, cs2))

    return waitq_pctile


def _ggm_waitq_pctile_whitt_wrap(t, p, arr_rate, svc_rate, c, ca2, cs2):
    return ggm_qwait_cdf_whitt(t, arr_rate, svc_rate, c, ca2, cs2) - p


def _ggm_qsize_prob_gt_0_whitt_5_2(arr_rate, svc_rate, c, ca2, cs2):
    """
    Return the approximate P(Q>0) in G/G/m queue using Whitt's simple
    approximation involving rho and P(W>0).

    This approximation is exact for M/M/m and has strong theoretical
    support for GI/M/m. It's described by Whitt as "crude" but is
    "a useful quick approximation".

    See Section 5 of Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161. In
    particular, this is Equation 5.2.


    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        ~ P(Q > 0)

    """

    rho = arr_rate / (svc_rate * float(c))
    pdelay = ggm_prob_wait_whitt(arr_rate, svc_rate, c, ca2, cs2)

    prob_gt_0 = rho * pdelay

    return prob_gt_0


def _ggm_qsize_prob_gt_0_whitt_5_1(arr_rate, svc_rate, c, ca2, cs2):
    """
    Return the approximate P(Q>0) in G/G/m queue using Whitt's approximation
    which is based on an exact expression for P(Q>0) given the CDF's
    of an interarrival time and a waiting time .

    This approximation is exact for M/M/m and has strong theoretical
    support for GI/M/m - see Equation 5.1. It is preferred to the cruder
    approximation given in Equation 5.2 (see ggm_qsize_prob_gt_0_whitt_5_2).

    See Section 5 of Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161. In
    particular, this is Equation 5.1.


    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    cv2_svc_time : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        ~ P(Q > 0

    """

    rho = arr_rate / (svc_rate * float(c))
    pdelay = ggm_prob_wait_whitt(arr_rate, svc_rate, c, ca2, cs2)

    # TODO - implement Equation 5.1 of Whitt (1995)

    return 0


def ggm_qsize_whitt_cq2(arr_rate, svc_rate, m, ca2, cs2):
    """
    Return the approximate squared coefficient of queue size in G/G/m queue.

    See Whitt, Ward. "Approximations for the GI/G/m queue"
    Production and Operations Management 2, 2 (Spring 1993): 114-161.
    Equation 5.6.

    Parameters
    ----------
    arr_rate : float
        average arrival rate to queueing system
    svc_rate : float
        average service rate (each server). 1/svc_rate is mean service time.
    c : int
        number of servers
    ca2 : float
        squared coefficient of variation for inter-arrival time distribution
    cs2 : float
        squared coefficient of variation for service time distribution

    Returns
    -------
    float
        scv of number in queue

    """


    eq = ggm_mean_qsize_whitt(arr_rate, svc_rate, m, ca2, cs2)

    cw2 = ggm_qwait_whitt_cw2(arr_rate, svc_rate, m, ca2, cs2)

    cq2 = (1/eq) + cw2

    return cq2
