//! 仅测试 / 内部使用的前向模式双数（设计文档 §4.5.4）。
//!
//! 前向模式实现简单到"很难写错"，是反向模式的理想 oracle；同时服务于
//! IFT 模式的残差 Jacobian 逐列构造（设计文档 §4.3.3）。
//! 不是公开的前向模式 API 承诺（`test-oracle` feature，默认开启）。

use num_traits::{Num, One, Zero};
use std::iter::Sum;
use std::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};

/// 双数 `re + du·ε`，`ε² = 0`。`re` 是函数值，`du` 是沿种子方向的方向导数。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Dual {
    pub re: f64,
    pub du: f64,
}

impl Dual {
    pub fn new(re: f64, du: f64) -> Self {
        Dual { re, du }
    }

    pub fn constant(re: f64) -> Self {
        Dual { re, du: 0.0 }
    }

    /// 便利构造：以 `x` 为值、以 `1.0` 为导数种子（对 x 求导）。
    pub fn seed(x: f64) -> Self {
        Dual { re: x, du: 1.0 }
    }

    pub fn sin(self) -> Self {
        Dual {
            re: self.re.sin(),
            du: self.du * self.re.cos(),
        }
    }

    pub fn cos(self) -> Self {
        Dual {
            re: self.re.cos(),
            du: -self.du * self.re.sin(),
        }
    }

    pub fn tanh(self) -> Self {
        let t = self.re.tanh();
        Dual {
            re: t,
            du: self.du * (1.0 - t * t),
        }
    }

    pub fn exp(self) -> Self {
        let e = self.re.exp();
        Dual {
            re: e,
            du: self.du * e,
        }
    }

    /// 定义域 re > 0（与 [`f64::ln`] 一致）。
    pub fn ln(self) -> Self {
        Dual {
            re: self.re.ln(),
            du: self.du / self.re,
        }
    }

    /// 定义域 re >= 0；re = 0 时反向为 +inf（与库约定一致）。
    pub fn sqrt(self) -> Self {
        let s = self.re.sqrt();
        Dual {
            re: s,
            du: self.du * 0.5 / s,
        }
    }

    pub fn abs(self) -> Self {
        let sign = if self.re > 0.0 {
            1.0
        } else if self.re < 0.0 {
            -1.0
        } else {
            0.0
        };
        Dual {
            re: self.re.abs(),
            du: sign * self.du,
        }
    }

    /// a^b，a > 0。
    pub fn powf(self, other: Dual) -> Self {
        let re = self.re.powf(other.re);
        let du = re * (other.du * self.re.ln() + other.re * self.du / self.re);
        Dual { re, du }
    }

    pub fn powi(self, n: i32) -> Self {
        let du = (n as f64) * self.re.powi(n - 1) * self.du;
        Dual {
            re: self.re.powi(n),
            du,
        }
    }

    pub fn recip(self) -> Self {
        Dual {
            re: 1.0 / self.re,
            du: -self.du / (self.re * self.re),
        }
    }

    /// y = self（即 y），x = other；d/dt atan2(y, x) = (x·y' - y·x')/r²。
    pub fn atan2(self, other: Dual) -> Self {
        let r2 = self.re * self.re + other.re * other.re;
        let du = (other.re * self.du - self.re * other.du) / r2;
        Dual {
            re: self.re.atan2(other.re),
            du,
        }
    }

    pub fn asin(self) -> Self {
        Dual {
            re: self.re.asin(),
            du: self.du / (1.0 - self.re * self.re).sqrt(),
        }
    }

    pub fn acos(self) -> Self {
        Dual {
            re: self.re.acos(),
            du: -self.du / (1.0 - self.re * self.re).sqrt(),
        }
    }

    pub fn sigmoid(self) -> Self {
        let s = 1.0 / (1.0 + (-self.re).exp());
        Dual {
            re: s,
            du: self.du * s * (1.0 - s),
        }
    }
}

impl Add for Dual {
    type Output = Dual;
    fn add(self, o: Dual) -> Dual {
        Dual {
            re: self.re + o.re,
            du: self.du + o.du,
        }
    }
}
impl Sub for Dual {
    type Output = Dual;
    fn sub(self, o: Dual) -> Dual {
        Dual {
            re: self.re - o.re,
            du: self.du - o.du,
        }
    }
}
impl Mul for Dual {
    type Output = Dual;
    fn mul(self, o: Dual) -> Dual {
        Dual {
            re: self.re * o.re,
            du: self.du * o.re + self.re * o.du,
        }
    }
}
impl Div for Dual {
    type Output = Dual;
    fn div(self, o: Dual) -> Dual {
        let re = self.re / o.re;
        let du = (self.du * o.re - self.re * o.du) / (o.re * o.re);
        Dual { re, du }
    }
}
impl Rem for Dual {
    type Output = Dual;
    fn rem(self, o: Dual) -> Dual {
        Dual {
            re: self.re % o.re,
            du: self.du,
        }
    }
}
impl Neg for Dual {
    type Output = Dual;
    fn neg(self) -> Dual {
        Dual {
            re: -self.re,
            du: -self.du,
        }
    }
}
impl AddAssign for Dual {
    fn add_assign(&mut self, o: Dual) {
        *self = *self + o;
    }
}
impl SubAssign for Dual {
    fn sub_assign(&mut self, o: Dual) {
        *self = *self - o;
    }
}
impl MulAssign for Dual {
    fn mul_assign(&mut self, o: Dual) {
        *self = *self * o;
    }
}
impl DivAssign for Dual {
    fn div_assign(&mut self, o: Dual) {
        *self = *self / o;
    }
}
impl RemAssign for Dual {
    fn rem_assign(&mut self, o: Dual) {
        *self = *self % o;
    }
}
impl Sum for Dual {
    fn sum<I: Iterator<Item = Dual>>(iter: I) -> Dual {
        iter.fold(Dual::constant(0.0), |a, b| a + b)
    }
}

impl Zero for Dual {
    fn zero() -> Self {
        Dual::constant(0.0)
    }
    fn is_zero(&self) -> bool {
        self.re == 0.0 && self.du == 0.0
    }
}
impl One for Dual {
    fn one() -> Self {
        Dual::constant(1.0)
    }
}
impl Num for Dual {
    type FromStrRadixErr = num_traits::ParseFloatError;
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        <f64 as Num>::from_str_radix(str, radix).map(Dual::constant)
    }
}
