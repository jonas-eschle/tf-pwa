----------------
Phase Space
----------------

:math:`N` body decay phase space can be defined as

.. math::
   \mathrm{d} \Phi(P;p_1,\cdots,p_n) = (2\pi)^4\delta^4(P - \sum {p_i}) \prod \theta(p^0)2\pi\delta(p_i^2 - m_i^2)\frac{\mathrm{d}^4 p_i}{(2\pi)^{4}}

or integrate :math:`p^0` as

.. math::
   \mathrm{d} \Phi(P;p_1,\cdots,p_n) = (2\pi)^4\delta^4(P - \sum {p_i}) \prod \frac{1}{2E_i}\frac{\mathrm{d}^3 \vec{p_i}}{(2\pi)^{3}}

by using the property of :math:`\delta`-function,

.. math::
   \delta(f(x)) = \sum_{i}\frac{\delta(x-x_i)}{|f'(x_i)|}

where :math:`x_i` is the root of :math:`f(x)=0`.

Phase space has the following chain rule,

.. math::
   \mathrm{d} \Phi(P;p_1,\cdots,p_n) =& (2\pi)^4\delta^4(P - \sum {p_i}) \prod \frac{1}{2E_i}\frac{\mathrm{d}^3 \vec{p_i}}{(2\pi)^{3}}  \\
   =& (2\pi)^4\delta^4(P - \sum_{i=0}^{m} {p_i} -q) \prod_{i=0}^{m} \frac{1}{2E_i}\frac{\mathrm{d}^3 \vec{p_i}}{(2\pi)^{3}} \prod_{i=m+1}^{n} \frac{1}{2E_i}\frac{\mathrm{d}^3 \vec{p_i}}{(2\pi)^{3}}\\
    & (2\pi)^4\delta^4(q - \sum_{i=m+1}^{n} {p_i})\frac{\mathrm{d}^4 q}{(2\pi)^4}\delta(q^2 - (\sum_{i=m+1}^{n} {p_i})^2)\mathrm{d} q^2 \\
   =& \mathrm{d}\Phi(P;p_1,\cdots,p_m,q)\frac{\mathrm{d} q^2}{2\pi}\mathrm{d}\Phi(q;p_{m+1},\cdots p_{n}) \label{chain_decay},

where :math:`q = \sum_{i=m+1}^{n}p_i`
is the invariant mass of particles :math:`m+1,\cdots,n`.


The two body decay is simple in the center mass frame :math:`P=(M,0,0,0)`,

.. math::
   \mathrm{d} \Phi(P;p_1,p_2) =& (2\pi)^4\delta^4(P - p_1 - p_2) \frac{1}{2E_1}\frac{\mathrm{d}^3 \vec{p_1}}{(2\pi)^{3}} \frac{1}{2E_2}\frac{\mathrm{d}^3 \vec{p_2}}{(2\pi)^{3}} \\
   =& 2\pi\delta(M - E_1 - E_2) \frac{1}{2E_1 }\frac{1}{2E_2}\frac{\mathrm{d}^3 \vec{p_2}}{(2\pi)^{3}} \\
   =& 2\pi\delta(M - \sqrt{|\vec{p}|^2 + m_1^2} - \sqrt{|\vec{p}|^2 + m_2^2}) \frac{1}{2E_1 }\frac{|\vec{p}|^2}{2E_2}\frac{\mathrm{d} |\vec{p}| \mathrm{d} \Omega}{(2\pi)^{3}} \\
   =& \frac{|\vec{p}|}{16 \pi^2 M} \mathrm{d} \Omega

where :math:`\mathrm{d} \Omega = \mathrm{d}(\cos\theta)\mathrm{d}\varphi` and

.. math::
   E_1 = \frac{M^2 + m_1^2 - m_2^2 }{2M} , E_1 = \frac{M^2 - m_1^2 + m_2^2 }{2M}

.. math::
   |\vec{p}| = \frac{\sqrt{(M^2 - (m_1 + m_2)^2)(M^2 -(m_1 - m_2)^2)}}{2M}


The three body decay in the center of mass frame :math:`P=(M,0,0,0),q^\star=(m_{23},0,0,0)`,

.. math::
   \mathrm{d} \Phi(P;p_1,p_2,p_3) =& \mathrm{d}\Phi(P;p_1,q) \mathrm{d}\Phi(q^\star;p_2^\star,p_3^\star) \frac{\mathrm{d} q^2}{2\pi} \\
   =& \frac{|\vec{p_1}||\vec{p_2^\star}|}{(2\pi)^5 16 M m_{23}} \mathrm{d} m_{23}^2 \mathrm{d} \Omega_1 \mathrm{d}\Omega_2^\star \\
   =& \frac{|\vec{p_1}||\vec{p_2^\star}|}{(2\pi)^5 8 M} \mathrm{d} m_{23} \mathrm{d} \Omega_1 \mathrm{d}\Omega_2^\star

The N body decay in the center of mass frame :math:`P=(M,0,0,0)`,

.. math::
   \mathrm{d} \Phi(P;p_1,\cdots,p_n) =& \mathrm{d}\Phi(P;p_1,q_1)\prod_{i=1}^{n-2} \frac{\mathrm{d} q_{i}^2}{2\pi}\mathrm{d}\Phi(q_{i},p_{i+1},p_{i+2})\\
   =& \frac{1}{2^{2n-2}(2\pi)^{3n-4}}\frac{|\vec{p_{1}}|}{M} \mathrm{d}\Omega_{1} \prod_{i=1}^{n-2} \frac{|\vec{p_{i+1}^\star}|}{M_{i}} \mathrm{d} M_{i}^2 \mathrm{d}\Omega_{i+1}^\star \\
   =& \frac{1}{2^n (2\pi)^{3n-4}}\frac{|\vec{p_{1}}|}{M} \mathrm{d}\Omega_{1} \prod_{i=1}^{n-2} |\vec{p_{i+1}^\star}| \mathrm{d} M_{i} \mathrm{d}\Omega_{i+1}^\star

where

.. math::
   M_{i}^2 = (\sum_{j > i} p_j)^2 ,\ |\vec{p_{i}^\star}| = \frac{\sqrt{(M_i^2 - (M_{i+1} + m_{i+1})^2)(M_i^2 - (M_{i+1} - m_{i+1})^2)}}{2 M_i}

with those limit

.. math::
   \sum_{j>i} m_{j} < M_{i+1} + m_{i+1} < M_{i} < M_{i-1} - m_{i} < M - \sum_{j \leq i }  m_i

Phase Space Generator
---------------------

For N body phase space

.. math::
   \mathrm{d} \Phi(P;p_1,\cdots,p_n) =
   \frac{1}{2^n (2\pi)^{3n-4}} \left( \frac{1}{M}\prod_{i=0}^{n-2}|\vec{p_{i+1}^\star}|  \right)\prod_{i=1}^{n-2} \mathrm{d} M_{i} \prod_{i=0}^{n-2} \mathrm{d}\Omega_{i+1}^\star,

take a weeker condition

.. math::
   \sum_{j>i} m_{j} < M_{i} < M - \sum_{j \leq i }  m_j,

has the simple limit at the factor term

.. math::
   \frac{1}{M}\prod_{i=0}^{n-2}|\vec{p_{i+1}^\star}|
      =&  \frac{1}{M}\prod_{i=0}^{n-2}q(M_i,M_{i+1},m_{i+1}) \\
      <&  \frac{1}{M}\prod_{i=0}^{n-2}q(max(M_i),min(M_{i+1}),m_{i+1})


* 1. Generate :math:`M_i` with the factor
* 2. Generate :math:`\mathrm{d}\Omega = \mathrm{d}\cos\theta \mathrm{d}\varphi`
* 3. boost :math:`p^\star=(\sqrt{|\vec{p*}|^2 + m^2} ,|\vec{p^\star}|\cos\theta\cos\varphi,|\vec{p^\star}|\sin\theta\sin\varphi,|\vec{p^\star}|\cos\theta,)`  to a same farme.
