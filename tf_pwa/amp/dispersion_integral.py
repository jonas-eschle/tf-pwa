import numpy as np
import tensorflow as tf

from tf_pwa.amp.core import Particle, register_particle
from tf_pwa.data import data_to_numpy


def build_integral(fun, s_range, s_th, N=1001, add_tail=True, method="tf"):
    """

    .. math::
       F(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    if method == "scipy":
        return build_integral_scipy(fun, s_range, s_th, N=N, add_tail=add_tail)
    else:
        return build_integral_tf(fun, s_range, s_th, N=N, add_tail=add_tail)


def build_integral_scipy(
    fun, s_range, s_th, N=1001, add_tail=True, _epsilon=1e-6
):
    """

    .. math::
       F(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    from scipy.integrate import quad

    s_min, s_max = s_range
    x = np.linspace(s_min, s_max, N)

    ret = []

    def f(s):
        with tf.device("CPU"):
            y = data_to_numpy(fun(s))
        return y

    for xi in x:
        if xi < s_th:
            y, e = quad(lambda s: f(s) / (s - xi), s_th + _epsilon, np.inf)
        else:
            y1, e1 = quad(lambda s: f(s) / (s - xi), s_th, xi - _epsilon)
            y2, e2 = quad(
                lambda s: f(s) / (s - xi), xi + _epsilon, s_max + 0.1
            )
            y3, e2 = quad(lambda s: f(s) / (s - xi), s_max + 0.1, np.inf)
            y = y1 + y2 + y3
        ret.append(y)
    return x, np.stack(ret)


def build_integral_tf(fun, s_range, s_th, N=1001, add_tail=True):
    """

    .. math::
       I(s) = P\\int_{s_{th}}^{\\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'
            = \\int_{s_{th}}^{s- \\epsilon} \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s + \\epsilon}^{s_{max} } \\frac{f(s')}{s'-s} \\mathrm{d} s' + \\int_{s_{max}}^{\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s'

    It require same :math:`\\epsilon` for :math:`s- \\epsilon` and :math:`s- \\epsilon` to get the Cauchy Principal Value. We used bin center to to keep the same :math:`\\epsilon` from left and right bound.

    """
    s_min, s_max = s_range
    delta = (s_min - s_max) / (N - 1)
    x = np.linspace(s_min - delta / 2, s_max + delta / 2, N + 1)
    shift = (s_th - s_min) % delta
    x = x + shift * delta
    x_center = (x[1:] + x[:-1]) / 2
    int_x = x[x > s_th + delta / 4]
    fx = fun(int_x) / (int_x - x_center[:, None])
    int_f = tf.reduce_mean(fx, axis=-1) * (int_x[-1] - int_x[0] + delta)
    if add_tail:
        int_f = int_f + build_integral_tail(
            fun, x_center, x[-1] + delta / 2, s_th, N
        )
    return x_center, int_f


def build_integral_tail(fun, x_center, tail, s_th, N=1001, _epsilon=1e-9):
    """

    Integration of the tail parts using tan transfrom.

    .. math::
       \\int_{s_{max}}^{\infty} \\frac{f(s')}{s'-s} \\mathrm{d} s' = \\int_{\\arctan s_{max}}^{\\frac{\\pi}{2}} \\frac{f(\\tan x)}{\\tan x-s} \\frac{\\mathrm{d} \\tan x}{\\mathrm{d} x} \\mathrm{d} x

    """
    x_min, x_max = np.arctan(tail), np.pi / 2
    delta = (x_min - x_max) / N
    x = np.linspace(x_min + delta / 2, x_max - delta / 2, N)
    tanx = tf.tan(x)
    dtanxdx = 1 / tf.cos(x) ** 2
    fx = fun(tanx) / (tanx - x_center[:, None]) * dtanxdx
    int_f = tf.reduce_mean(fx, axis=-1) * (np.pi / 2 - np.arctan(tail))
    return int_f


def complex_q(s, m1, m2):
    q2 = (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / (4 * s)
    q = tf.sqrt(tf.complex(q2, tf.zeros_like(q2)))
    return q


def chew_mandelstam(m, m1, m2):
    """
    Chew-Mandelstam function
    """
    s = m * m
    C = lambda x: tf.complex(x, tf.zeros_like(x))
    m1 = tf.cast(m1, s.dtype)
    m2 = tf.cast(m2, s.dtype)
    q = complex_q(s, m1, m2)
    s1 = m1 * m1
    s2 = m2 * m2
    a = (
        C(2 / m)
        * q
        * tf.math.log((C(s1 + s2 - s) + C(2 * m) * q) / C(2 * m1 * m2))
    )
    b = (s1 - s2) * (1 / s - 1 / (m1 + m2) ** 2) * tf.math.log(m1 / m2)
    ret = a - C(b)
    return ret / (16 * np.pi**2)


class LinearInterpFunction:
    def __init__(self, x_range, y):
        x_min, x_max = x_range
        N = y.shape[-1]
        self.x_min = x_min
        self.x_max = x_max
        self.delta = (self.x_max - self.x_min) / (N - 1)
        self.N = N
        self.y = y

    def __call__(self, x):
        diff = (x - self.x_min) / self.delta
        idx0 = diff // 1.0
        idx = tf.cast(idx0, tf.int32)
        left = tf.gather(self.y, idx)
        right = tf.gather(self.y, idx + 1)
        k = diff - idx0
        return (1 - k) * left + k * right


class DispersionIntegralFunction(LinearInterpFunction):
    def __init__(self, fun, s_range, s_th, N=1001, method="tf"):
        self.fun = fun
        x_center, y = build_integral(fun, s_range, s_th, N, method=method)
        super().__init__((x_center[0], x_center[-1]), y)


@register_particle("DI")
class DispersionIntegralParticle(Particle):
    """

    "DI" (Dispersion Integral) model is the model used in `PRD78,074023(2008) <https://inspirehep.net/literature/793474>`_ . In the model a linear interpolation is used to avoid integration every times in  fitting. No paramters are allowed in the integration, unless `dyn_int=True`.

    .. math::
        f(s) = \\frac{1}{m_0^2 - s - \\sum_{i} [Re \\Pi_i(s) - Re\\Pi_i(m_0^2)] - i \\sum_{i} \\rho'_i(s) }

    where :math:`\\rho'_i(s) = g_i^2 \\rho_i(s) F_i^2(s)` is the phase space with barrier factor :math:`F_i^2(s)=\\exp(-\\alpha k_i^2)`.

    The real parts of :math:`\Pi(s)` is defined using the dispersion intergral

    .. math::

        Re \\Pi_i(s) = \\frac{1}{\\pi} P \\int_{s_{th,i}}^{\\infty} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s' = \\lim_{\\epsilon \\rightarrow 0} \\left[ \\int_{s_{th,i}}^{s-\\epsilon} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s' +\\int_{s+\\epsilon}^{\\infty} \\frac{\\rho'_i(s')}{s' - s} \\mathrm{d} s'\\right]

    The reprodution of the Fig1 in  `PRD78,074023(2008) <https://inspirehep.net/literature/793474>`_ .

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> from tf_pwa.amp.dispersion_integral import DispersionIntegralParticle
        >>> plt.clf()
        >>> m = np.linspace(0.6, 1.6, 1001)
        >>> s = m * m
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> p = DispersionIntegralParticle("A0_980_DI", mass_range=(0,2.0), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101)
        >>> p.init_params()
        >>> y1 = p.rho_prime(s, *p.mass_list[0])
        >>> scale1 = 1/np.max(y1)
        >>> x1 = p.int_f[0](s)/np.pi
        >>> p.alpha = 2.5
        >>> p.init_integral()
        >>> y2 = p.rho_prime(s, *p.mass_list[0])
        >>> scale2 = 1/np.max(y2)
        >>> x2 = p.int_f[0](s)/np.pi
        >>> p_ref = DispersionIntegralParticle("A0_980_DI", mass_range=(0.6,1.6), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=11, int_method="scipy")
        >>> p_ref.init_params()
        >>> s_ref = np.linspace(0.6**2, 1.6**2-1e-6, 11)
        >>> x_ref = p_ref.int_f[0](s_ref)/np.pi
        >>> _ = plt.plot(m, y1* scale1, label="$\\\\rho'(s)$")
        >>> _ = plt.plot(m, x1* scale1, label="Re $\\\\Pi (s)$")
        >>> _ = plt.plot(m, y2* scale2, linestyle="--")
        >>> _ = plt.plot(m, x2* scale2, linestyle="--")
        >>> _ = plt.scatter(np.sqrt(s_ref), x_ref* scale1, label="scipy integration")
        >>> _ = plt.legend()

    The Argand plot

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> from tf_pwa.utils import plot_particle_model
        >>> _ = plot_particle_model("DI", dict(mass=0.98, mass_range=(0,2.0), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05])

    """

    def __init__(
        self,
        *args,
        mass_range=(0, 5),
        mass_list=[[0.493677, 0.493677]],
        l_list=None,
        int_N=1001,
        int_method="tf",
        dyn_int=False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mass_range = mass_range
        self.srange = (mass_range[0] ** 2, mass_range[1] ** 2)
        self.mass_list = mass_list
        self.int_N = int_N
        self.int_method = int_method
        if l_list is None:
            l_list = [0] * len(mass_list)
        self.l_list = l_list
        self.dyn_int = dyn_int

    def init_params(self):
        super().init_params()
        self.alpha = 2.0
        self.gi = []
        for idx, _ in enumerate(self.mass_list):
            name = f"g_{idx}"
            self.gi.append(self.add_var(name))
        self.init_integral()

    def init_integral(self):
        self.int_f = []
        for idx, ((m1, m2), l) in enumerate(zip(self.mass_list, self.l_list)):
            fi = lambda s: self.rho_prime(s, m1, m2, l)
            int_fi = DispersionIntegralFunction(
                fi,
                self.srange,
                (m1 + m2) ** 2,
                N=self.int_N,
                method=self.int_method,
            )
            self.int_f.append(int_fi)

    def q2_ch(self, s, m1, m2):
        return (s - (m1 + m2) ** 2) * (s - (m1 - m2) ** 2) / s / 4

    def im_weight(self, s, m1, m2, l=0):
        return tf.ones_like(s)

    def rho_prime(self, s, m1, m2, l=0):
        q2 = self.q2_ch(s, m1, m2)
        q2 = tf.where(s > (m1 + m2) ** 2, q2, tf.zeros_like(q2))
        rho = 2 * tf.sqrt(q2 / s)
        F = tf.exp(-self.alpha * q2)
        return rho * F**2 / self.im_weight(s, m1, m2, l)

    def __call__(self, m):
        if self.dyn_int:
            self.init_integral()
        s = m * m
        m0 = self.get_mass()
        s0 = m0 * m0
        gi = tf.stack([var() ** 2 for var in self.gi], axis=-1)
        ims = []
        res = []
        for (m1, m2), li, f in zip(self.mass_list, self.l_list, self.int_f):
            w = self.im_weight(s, m1, m2, li)
            w0 = self.im_weight(s0, m1, m2, li)
            tmp_i = self.rho_prime(s, m1, m2, li) * w
            tmp_r = f(s) * w - f(s0) * w0
            ims.append(tmp_i)
            res.append(tmp_r)
        im = tf.stack(ims, axis=-1)
        re = tf.stack(res, axis=-1)
        real = s0 - s - tf.reduce_sum(gi * re, axis=-1) / np.pi
        imag = tf.reduce_sum(gi * im, axis=-1)
        dom = real**2 + imag**2
        return tf.complex(real / dom, imag / dom)

    def get_amp(self, *args, **kwargs):
        return self(args[0]["m"])


@register_particle("DI2")
class DispersionIntegralParticle2(DispersionIntegralParticle):
    """

    Dispersion Integral model. In the model a linear interpolation is used to avoid integration every times in fitting. No paramters are allow in the integration.

    .. math::
        f(s) = \\frac{1}{m_0^2 - s - \\sum_{i} g_i^2 [Re\\Pi_i(s) -Re\\Pi_i(m_0^2) + i Im \\Pi_i(s)] }

    where :math:`Im \\Pi_i(s)=\\rho_i(s)n_i^2(s)`, :math:`n_i(s)={q}^{l} {B_l'}(q,1/d, d)`.

    The real parts of :math:`\Pi(s)` is defined using the dispersion intergral

    .. math::

        Re \\Pi_i(s) = \\frac{\\color{red}(s-s_{th,i})}{\\pi} P \\int_{s_{th,i}}^{\\infty} \\frac{Im \\Pi_i(s')}{(s' - s)({\\color{red} s'-s_{th,i} })} \\mathrm{d} s'

    .. note::

        Small `int_N` will have bad precision.

    The shape of :math:`\\Pi(s)` and comparing to Chew-Mandelstam function :math:`\\Sigma(s)`

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import tensorflow as tf
        >>> from tf_pwa.amp.dispersion_integral import DispersionIntegralParticle2, chew_mandelstam
        >>> plt.clf()
        >>> m = np.linspace(0.6, 1.6, 1001)
        >>> s = m * m
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> p = DispersionIntegralParticle2("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101)
        >>> p.init_params()
        >>> p2 = DispersionIntegralParticle2("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=1001)
        >>> p2.init_params()
        >>> y1 = p.rho_prime(s, M_K, M_K)* (s-M_K**2*4)
        >>> scale1 = 1/np.max(y1)
        >>> x1 = p.int_f[0](s) * (s-M_K**2*4)/np.pi
        >>> x2 = p2.int_f[0](s) * (s-M_K**2*4)/np.pi
        >>> p_ref = DispersionIntegralParticle2("A0_980_DI", mass_range=(0.6,1.6), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=11, int_method="scipy")
        >>> p_ref.init_params()
        >>> s_ref = np.linspace(0.6**2, 1.6**2-1e-6, 11)
        >>> x_ref = p_ref.int_f[0](s_ref) * (s_ref-M_K**2*4)/np.pi
        >>> x_chew = chew_mandelstam(s, M_K, M_K) * np.pi**2 * 4
        >>> _ = plt.plot(m, x1* scale1, label="Re $\\\\Pi (s), N=101$")
        >>> _ = plt.plot(m, x2* scale1, label="Re $\\\\Pi (s), N=1001$")
        >>> _ = plt.plot(m, tf.math.real(x_chew).numpy() * scale1, label="$Re\\\\Sigma(s)$")
        >>> _ = plt.plot(m, tf.math.imag(x_chew).numpy() * scale1, label="$Im \\\\Sigma(s)$")
        >>> _ = plt.plot(m, y1* scale1, label="Im $\\\\Pi (s)$")
        >>> _ = plt.scatter(np.sqrt(s_ref), x_ref* scale1, label="scipy integration")
        >>> _ = plt.legend()

    The Argand plot

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("DI2", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,1], int_N=101), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05])
        >>> _ = plot_particle_model("DI2", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,1], int_N=1001), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05], axis=axis)
        >>> _ = axis[3].legend(["N=101", "N=1001"])


    """

    def rho_prime(self, s, m1, m2, l=0):
        q2 = self.q2_ch(s, m1, m2)
        q2 = tf.where(s > (m1 + m2) ** 2, q2, tf.zeros_like(q2))
        rho = 2 * tf.sqrt(q2 / s)
        from tf_pwa.breit_wigner import Bprime_q2

        rhop = rho * q2**l * Bprime_q2(l, q2, 1 / self.d**2, self.d) ** 2
        return rhop / self.im_weight(s, m1, m2, l)

    def im_weight(self, s, m1, m2, l=0):
        return s - (m1 + m2) ** 2


@register_particle("DI3")
class DispersionIntegralParticle3(DispersionIntegralParticle2):
    """

    Dispersion Integral model. In the model a linear interpolation is used to avoid integration every times in fitting. No paramters are allow in the integration.

    .. math::
        f(s) = \\frac{1}{m_0^2 - s - \\sum_{i} g_i^2 [Re\\Pi_i(s) -Re\\Pi_i(m_0^2) + i Im \\Pi_i(s)] }

    where :math:`Im \\Pi_i(s)=\\rho_i(s)n_i^2(s)`, :math:`n_i(s)={q}^{l} {B_l'}(q,1/d, d)`.

    The real parts of :math:`\Pi(s)` is defined using the dispersion intergral

    .. math::

        Re \\Pi_i(s) = \\frac{\\color{red}s}{\\pi} P \\int_{s_{th,i}}^{\\infty} \\frac{Im \\Pi_i(s')}{(s' - s)({\\color{red}s'})} \\mathrm{d} s'

    .. note::

        Small `int_N` will have bad precision.

    The shape of :math:`\\Pi(s)`

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> import tensorflow as tf
        >>> from tf_pwa.amp.dispersion_integral import DispersionIntegralParticle3, chew_mandelstam
        >>> plt.clf()
        >>> m = np.linspace(0.6, 1.6, 1001)
        >>> s = m * m
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> p = DispersionIntegralParticle3("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=101)
        >>> p.init_params()
        >>> p2 = DispersionIntegralParticle3("A0_980_DI", mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=1001)
        >>> p2.init_params()
        >>> y1 = p.rho_prime(s, M_K, M_K)* (s)
        >>> scale1 = 1/np.max(y1)
        >>> x1 = p.int_f[0](s) * (s)/np.pi
        >>> x2 = p2.int_f[0](s) * (s)/np.pi
        >>> p_ref = DispersionIntegralParticle3("A0_980_DI", mass_range=(0.6,1.6), mass_list=[[M_K, M_K],[M_eta, M_pi]], int_N=11, int_method="scipy")
        >>> p_ref.init_params()
        >>> s_ref = np.linspace(0.6**2, 1.6**2-1e-6, 11)
        >>> x_ref = p_ref.int_f[0](s_ref) * (s_ref)/np.pi
        >>> _ = plt.plot(m, x1* scale1, label="Re $\\\\Pi (s), N=101$")
        >>> _ = plt.plot(m, x2* scale1, label="Re $\\\\Pi (s), N=1001$")
        >>> _ = plt.plot(m, y1* scale1, label="Im $\\\\Pi (s)$")
        >>> _ = plt.scatter(np.sqrt(s_ref), x_ref* scale1, label="scipy integration")
        >>> _ = plt.legend()

    The Argand plot

    .. plot::

        >>> import matplotlib.pyplot as plt
        >>> plt.clf()
        >>> M_K = 0.493677
        >>> M_eta = 0.547862
        >>> M_pi = 0.1349768
        >>> from tf_pwa.utils import plot_particle_model
        >>> axis = plot_particle_model("DI3", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,1], int_N=101, dyn_int=True), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05])
        >>> _ = plot_particle_model("DI3", dict(mass=0.98, mass_range=(0.5,1.7), mass_list=[[M_K, M_K],[M_eta, M_pi]], l_list=[0,1], int_N=1001), {"R_BC_g_0": 0.415,"R_BC_g_1": 0.405}, mrange=[0.93, 1.05], axis=axis)
        >>> _ = axis[3].legend(["N=101", "N=1001"])


    """

    def im_weight(self, s, m1, m2, l=0):
        return s
