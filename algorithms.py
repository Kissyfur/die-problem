import numpy as np


def to_beta(p):
    return np.log(p)


def to_ps(b):
    max_betas = np.max(b, axis=1)
    p_hat = np.exp(b.T - max_betas)
    p = p_hat / np.sum(p_hat, axis=0)
    return p.T


def to_p(beta):
    max_beta = np.max(beta)
    p_hat = np.exp(beta - max_beta)
    p = p_hat / np.sum(p_hat)
    return p


def gradient(beta, side):
    g = -to_p(beta)
    g[side] += 1
    g[0] = 0
    return g


def kl_divergence(beta_real, betas):
    p_real = to_p(beta_real)
    qs = to_ps(betas)

    non_zeros = np.nonzero(p_real)
    kl = -p_real[non_zeros] * np.log(qs.T[non_zeros]).T + p_real[non_zeros] * np.log(p_real)[non_zeros]
    kl = np.sum(kl, axis=1)
    return kl


class Algorithm(object):
    def __init__(self):
        self.name = ""
        self.betas_to_keep = 100

    def get_step(self, a, b, t):
        return a / (b + t)

    def learn(self, beta, hyperaparams, y_train):
        return None

    def adjust_hyperparams(self, beta, y_train, real_beta):
        best_kl = 1e99
        a_exps = np.arange(-2, 8)
        y_train_completed = y_train
        b_exps = np.arange(-5, 9)
        hyperparams = {}
        table = np.ones((len(a_exps) + 1, len(b_exps) + 1)) * (-1)

        for i in range(len(a_exps)):
            a = np.power(10., a_exps[i])
            hyperparams["a"] = a
            for j in range(len(b_exps)):
                b = np.power(10., b_exps[j])
                hyperparams["b"] = b
                try:
                    betas = self.learn(np.copy(beta), hyperparams, y_train)
                    eval_bound = max(-10, -len(y_train_completed))
                    kl = kl_divergence(real_beta, betas[eval_bound:])
                    kl = np.sum(kl)
                    if kl < best_kl:
                        best_a = a
                        best_b = b
                        best_kl = kl
                except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
                    pass
        if (best_a == 10. ** a_exps[0]):
            print("Decrease min a range")
            # exit()
        if (best_a == 10. ** a_exps[-1]):
            print("Increase max a range")
            # exit()
        if (best_b == 10. ** b_exps[0]):
            print("Decrease min b range")
            # exit()
        if (best_b == 10. ** b_exps[-1]):
            print("Increase max max  b range")
            # exit()
        # print("Best lr for",self.name,"is a:", best_a, "b:", best_b, "with log_p", best_logp )
        print("Best lr for", self.name, "is a:", best_a, "b:", best_b, "with kl", best_kl)

        return {"a": best_a, "b": best_b}


class SGD(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "SGD"

    def direction(self, beta, side):
        g = gradient(beta, side)
        return g

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        a = hyperparams["a"]
        b = hyperparams["b"]
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        it = 1
        betas[0] = beta.copy()
        for i in range(n):
            step = self.get_step(a, b, i)
            g = self.direction(beta, y_train[i])
            beta = beta + g * step

            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
            # print("t=",t, to_p(betas[t]))
        return betas


class ADAGRAD(SGD):
    def __init__(self):
        super().__init__()
        self.name = "AdaGrad"
        self.epsilon = 1e-8

    def get_step(self, a, G):
        return a / np.sqrt(G + self.epsilon)

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        a = hyperparams["a"]
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        it = 1
        G = np.zeros(len(beta))
        for i in range(n):
            g = self.direction(beta, y_train[i])
            G += g*g
            step = self.get_step(a, G)
            beta = beta + g * step
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas

    def adjust_hyperparams(self, beta, y_train, real_beta):
        # best_logp = -1e99
        best_kl = 1e99
        aopt = np.arange(0.01, 1., 0.1)
        # y_train_completed = np.hstack([y_train, np.arange(len(beta)),np.arange(len(beta))])
        y_train_completed = y_train
        hyperparams = {}

        for a in aopt:
            # a = np.power(10., a_exps[i])
            hyperparams["a"] = a

            try:
                betas = self.learn(np.copy(beta), hyperparams, y_train)
                eval_bound = max(-10, -len(y_train_completed))
                kl = kl_divergence(real_beta, betas[eval_bound:])
                kl = np.sum(kl)

                if kl < best_kl:
                    best_a = a
                    best_kl = kl

            except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
                pass
        # print("Best lr for",self.name,"is a:", best_a, "b:", best_b, "with log_p", best_logp )
        print("Best lr for", self.name, "is a:", best_a, "with kl", best_kl)

        return {"a": best_a}


class SNGD2(SGD):
    def __init__(self):
        super().__init__()
        self.name = "SNGD2"

    def direction(self, beta, side):
        g = SGD.direction(self, beta, side)
        p = to_p(beta)
        inv_p = (1. / p)
        H = np.diag(inv_p[1:]) + inv_p[0]
        He = np.zeros((len(beta), len(beta)))
        He[1:, 1:] = H
        ng = np.dot(He, g)
        ng[0] = 0
        return ng


class SNGD(SGD):
    def __init__(self):
        super().__init__()
        self.name = "SNGD"

    def direction(self, beta, side):
        p = to_p(beta)
        if side == 0:
            ng = -np.ones(len(beta)) / p[0]
            ng[0] = 0
            return ng
        else:
            ng = np.zeros(len(beta))
            ng[side] = 1. / p[side]
            return ng


class MOD(Algorithm):
    def __init__(self, eras=100):
        super().__init__()
        self.name = "MOD" + str(eras)
        self.eras = eras

    def get_inv_matrix(self, beta):
        p = to_p(beta)
        inv_p = (1. / p)
        H = np.diag(inv_p[1:]) + inv_p[0]
        He = np.zeros((len(beta), len(beta)))
        He[1:, 1:] = H
        return He

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        it = 1
        a = hyperparams["a"]
        b = hyperparams["b"]
        H = None
        for i in range(n):
            step = self.get_step(a, b, i)
            if i % self.eras == 0:
                H = self.get_inv_matrix(beta)
            g = gradient(beta, y_train[i])
            ng = np.dot(H, g)
            beta = beta + ng * step
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas


class MEGD(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "MEGD"

    def get_inv_matrix(self, beta):
        p = to_p(beta)
        inv_p = (1. / p)
        H = np.diag(inv_p[1:]) + inv_p[0]
        He = np.zeros((len(beta), len(beta)))
        He[1:, 1:] = H
        return He

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        it = 1
        a = hyperparams["a"]
        b = hyperparams["b"]
        H = self.get_inv_matrix(beta)
        for i in range(n):
            step = self.get_step(a, b, i)
            g = gradient(beta, y_train[i])
            ng = np.dot(H, g)
            beta = beta + ng * step
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas


class DSNGD(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "DSNGD"

    def natural_gradient_approximation(self, beta, beta_naturalizer, side):
        ng = np.zeros(len(beta))
        g = gradient(beta, side)
        p = to_p(beta_naturalizer)
        inv_p = (1. / p)
        ng[1:] = inv_p[1:] * g[1:] + inv_p[0] * np.sum(g[1:])
        return ng

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        # self.H = np.zeros((len(beta) - 1, len(beta) - 1))
        # self.ng = np.zeros(len(beta))
        # self.d = np.diag_indices(len(beta) - 1)
        a = hyperparams["a"]
        b = hyperparams["b"]
        a_nat = 1000.
        b_nat = 1000.
        # theta = np.ones(len(beta))*1
        beta_nat = np.zeros(len(beta))
        it = 1
        sgd = SGD()
        for i in range(n):
            step = self.get_step(a, b, i)
            ng = self.natural_gradient_approximation(beta, beta_nat, y_train[i])
            beta = beta + ng * step

            step_nat = self.get_step(a_nat, b_nat, i)
            g = sgd.direction(beta_nat, y_train[i])
            beta_nat = beta_nat + g * step_nat
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas


class CSNGD(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "CSNGD"

    def natural_gradient_approximation(self, beta, beta_naturalizer, side):
        ng = np.zeros(len(beta))
        g = gradient(beta, side)
        p = to_p(beta_naturalizer)
        inv_p = (1. / p)
        ng[1:] = inv_p[1:] * g[1:] + inv_p[0] * np.sum(g[1:])

        # p = to_p(beta)
        # p_nat = to_p(beta_naturalizer)
        #
        # p_nat_inv = (1./p_nat)
        # ng = -p * p_nat_inv
        # ng = ng - ng[0] + p_nat_inv[side]
        # if side == 0:
        #     ng = ng - p_nat_inv[0]
        # ng[0] = 0
        return ng


    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        a = hyperparams["a"]
        b = hyperparams["b"]
        a_nat = 1000.
        b_nat = 1000.
        beta_nat = np.zeros(len(beta))
        it = 1
        sgd = SGD()
        for i in range(n):
            step = self.get_step(a, b, i)
            ng = self.natural_gradient_approximation(beta, beta_nat, y_train[i])
            beta = beta + ng * step
            step_nat = self.get_step(a_nat, b_nat, i)
            g = sgd.direction(beta_nat, y_train[i])
            beta_nat = beta_nat + g * step_nat
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas


class CSNGD2(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "CSNGD2"

    def get_inv_matrix(self, beta):
        p = to_p(beta)
        inv_p = (1. / p)
        H = np.diag(inv_p[1:]) + inv_p[0]
        He = np.zeros((len(beta), len(beta)))
        He[1:, 1:] = H
        return He

    def natural_gradient_approximation(self, beta, beta_naturalizer, side):
        g = gradient(beta, side)
        G = self.get_inv_matrix(beta_naturalizer)
        # ng = np.zeros(len(g))
        ng = np.dot(G, g)
        ng[0] = 0
        return ng

    def learn(self, beta, hyperparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        a = hyperparams["a"]
        b = hyperparams["b"]
        a_nat = 1000.
        b_nat = 1000.
        beta_nat = np.zeros(len(beta))
        it = 1
        sgd = SGD()
        for i in range(n):
            step = self.get_step(a, b, i)
            ng = self.natural_gradient_approximation(beta, beta_nat, y_train[i])
            beta = beta + ng * step
            step_nat = self.get_step(a_nat, b_nat, i)
            g = sgd.direction(beta_nat, y_train[i])
            beta_nat = beta_nat + g * step_nat
            if ((i+1) % length) == 0:
                betas[it] = beta
                it = it + 1
        return betas


class DSNGD_map(DSNGD):
    def __init__(self):
        DSNGD.__init__(self)
        self.name = "DSNGD_map"

    def learn(self, beta, hyperparams, y_train):
        a = hyperparams["a"]
        b = hyperparams["b"]
        theta = np.ones(len(beta)) * 1
        betas = np.zeros((len(y_train) + 1, len(beta)))
        betas[0] = beta
        t = 1
        for y in y_train:
            ntheta = theta / np.sum(theta)
            beta_nat = to_beta(ntheta)
            # print("theta:",theta, "ntheta:",ntheta, "beta_gradient": beta_gradient, "beta)
            step = self.get_step(a, b, t)
            betas[t] = self.follow_line(betas[t - 1], beta_nat, y, step)
            theta[y] += 1
            t += 1
        return betas


class MAP(Algorithm):
    def __init__(self):
        super().__init__()
        self.name = "MAP"

    def learn(self, beta, hyperaparams, y_train):
        n = len(y_train)
        m = min(self.betas_to_keep, n)
        length = int(n / m)
        betas = np.zeros((self.betas_to_keep + 1, len(beta)))
        betas[0] = beta.copy()
        a = hyperaparams["a"]
        theta = np.ones(len(beta)) * a
        it = 1
        for i in range(n):
            theta[y_train[i]] += 1
            ntheta = theta / np.sum(theta)
            if ((i+1) % length) == 0:
                betas[it] = to_beta(ntheta)
                it = it + 1
        return betas

    def adjust_hyperparams(self, beta, y_train, real_beta):
        best_kl = 1e99
        a_exps = np.arange(-50, 6)
        y_train_completed = y_train
        hyperparams = {}
        for a in np.power(10., a_exps):
            try:
                hyperparams["a"] = a
                betas = self.learn(np.copy(beta), hyperparams, y_train)
                eval_bound = max(-10, -len(y_train_completed))
                kl = kl_divergence(real_beta, betas[eval_bound:])
                kl = np.sum(kl)
                if kl < best_kl:
                    best_a = a
                    best_kl = kl
            except (OverflowError, np.linalg.linalg.LinAlgError, FloatingPointError):
                pass
        print("Best lr for", self.name, "is a:", best_a, "with kl", best_kl)
        return {"a": best_a}
