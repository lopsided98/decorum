use core::cmp::Ordering;
use core::convert::{Infallible, TryFrom};
use core::fmt::Debug;
use core::ops::{Add, Div, Mul, Rem, Sub};
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{ControlFlow, FromResidual, Try};

use crate::cmp::UndefinedError;
use crate::constraint::{Constraint, ExpectConstrained as _};
use crate::proxy::{ClosedProxy, ErrorOf, ExpressionOf, Proxy};
use crate::{
    with_binary_operations, with_primitives, BinaryReal, Codomain, Float, Infinite, Primitive,
    UnaryReal,
};

pub use crate::proxy::BranchOf;

pub use Expression::Defined;
pub use Expression::Undefined;

#[macro_export]
macro_rules! expression {
    ($x:expr) => {
        match $x {
            Expression::Defined(inner) => inner,
            _ => {
                return $x;
            }
        }
    };
}

pub trait Divergence {
    type Branch<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E>;

    fn from_residual<T, E>(residual: E) -> Self::Branch<T, E>
    where
        E: Debug;
}

impl Divergence for Infallible {
    type Branch<T, E> = T;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }

    fn from_residual<T, E>(_residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        unreachable!()
    }
}

pub trait NonResidual<T>: Divergence<Branch<T, ErrorOf<T>> = T>
where
    T: ClosedProxy,
{
}

impl<T, M> NonResidual<T> for M
where
    T: ClosedProxy,
    M: Divergence<Branch<T, ErrorOf<T>> = T>,
{
}

pub trait ResidualBranch {}

impl<T, E> ResidualBranch for Expression<T, E> {}

impl<T> ResidualBranch for Option<T> {}

impl<T, E> ResidualBranch for Result<T, E> {}

pub enum Assert {}

impl Divergence for Assert {
    type Branch<T, E> = T;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        output
    }

    fn from_residual<T, E>(residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Err(residual).expect_constrained()
    }
}

pub enum TryExpression {}

impl Divergence for TryExpression {
    type Branch<T, E> = Expression<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Defined(output)
    }

    fn from_residual<T, E>(residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Undefined(residual)
    }
}

pub enum TryOption {}

impl Divergence for TryOption {
    type Branch<T, E> = Option<T>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Some(output)
    }

    fn from_residual<T, E>(_residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        None
    }
}

pub enum TryResult {}

impl Divergence for TryResult {
    type Branch<T, E> = Result<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E> {
        Ok(output)
    }

    fn from_residual<T, E>(residual: E) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        Err(residual)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Expression<T, E = ()> {
    Defined(T),
    Undefined(E),
}

impl<T, E> Expression<T, E> {
    pub fn unwrap(self) -> T {
        match self {
            Defined(defined) => defined,
            _ => panic!(),
        }
    }

    pub fn map<U, F>(self, mut f: F) -> Expression<U, E>
    where
        F: FnMut(T) -> U,
    {
        match self {
            Defined(defined) => Defined(f(defined)),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn and_then<U, F>(self, mut f: F) -> Expression<U, E>
    where
        F: FnMut(T) -> Expression<U, E>,
    {
        match self {
            Defined(defined) => f(defined),
            Undefined(undefined) => Undefined(undefined),
        }
    }

    pub fn defined(self) -> Option<T> {
        match self {
            Defined(defined) => Some(defined),
            _ => None,
        }
    }

    pub fn undefined(self) -> Option<E> {
        match self {
            Undefined(undefined) => Some(undefined),
            _ => None,
        }
    }

    pub fn is_defined(&self) -> bool {
        matches!(self, Defined(_))
    }

    pub fn is_undefined(&self) -> bool {
        matches!(self, Undefined(_))
    }
}

impl<T, P> BinaryReal for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Clone + UndefinedError,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Self) -> Self::Superset {
        BinaryReal::div_euclid(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Self) -> Self::Superset {
        BinaryReal::rem_euclid(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Self) -> Self::Superset {
        BinaryReal::pow(expression!(self), expression!(n))
    }

    #[cfg(feature = "std")]
    fn log(self, base: Self) -> Self::Superset {
        BinaryReal::log(expression!(self), expression!(base))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Self) -> Self::Superset {
        BinaryReal::hypot(expression!(self), expression!(other))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Self) -> Self::Superset {
        BinaryReal::atan2(expression!(self), expression!(other))
    }
}

impl<T, P> BinaryReal<T> for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Clone + UndefinedError,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: T) -> Self::Superset {
        BinaryReal::div_euclid(expression!(self), expression!(Proxy::<T, P>::new(n)))
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: T) -> Self::Superset {
        BinaryReal::rem_euclid(expression!(self), expression!(Proxy::<T, P>::new(n)))
    }

    #[cfg(feature = "std")]
    fn pow(self, n: T) -> Self::Superset {
        BinaryReal::pow(expression!(self), expression!(Proxy::<T, P>::new(n)))
    }

    #[cfg(feature = "std")]
    fn log(self, base: T) -> Self::Superset {
        BinaryReal::log(expression!(self), expression!(Proxy::<T, P>::new(base)))
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: T) -> Self::Superset {
        BinaryReal::hypot(expression!(self), expression!(Proxy::<T, P>::new(other)))
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: T) -> Self::Superset {
        BinaryReal::atan2(expression!(self), expression!(Proxy::<T, P>::new(other)))
    }
}

impl<T, P> BinaryReal<Proxy<T, P>> for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Clone + UndefinedError,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    #[cfg(feature = "std")]
    fn div_euclid(self, n: Proxy<T, P>) -> Self::Superset {
        BinaryReal::div_euclid(expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn rem_euclid(self, n: Proxy<T, P>) -> Self::Superset {
        BinaryReal::rem_euclid(expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn pow(self, n: Proxy<T, P>) -> Self::Superset {
        BinaryReal::pow(expression!(self), n)
    }

    #[cfg(feature = "std")]
    fn log(self, base: Proxy<T, P>) -> Self::Superset {
        BinaryReal::log(expression!(self), base)
    }

    #[cfg(feature = "std")]
    fn hypot(self, other: Proxy<T, P>) -> Self::Superset {
        BinaryReal::hypot(expression!(self), other)
    }

    #[cfg(feature = "std")]
    fn atan2(self, other: Proxy<T, P>) -> Self::Superset {
        BinaryReal::atan2(expression!(self), other)
    }
}

impl<T, P> Codomain for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: UndefinedError,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    type Superset = Self;
}

impl<T, P> From<T> for Expression<Proxy<T, P>, ErrorOf<Proxy<T, P>>>
where
    T: Float + Primitive,
    P: Constraint,
{
    fn from(inner: T) -> Self {
        Proxy::try_new(inner).into()
    }
}

impl<T, P> From<Proxy<T, P>> for Expression<Proxy<T, P>, ErrorOf<Proxy<T, P>>>
where
    T: Float + Primitive,
    P: Constraint,
{
    fn from(proxy: Proxy<T, P>) -> Self {
        Defined(proxy)
    }
}

impl<T, E> From<Result<T, E>> for Expression<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(output) => Defined(output),
            Err(error) => Undefined(error),
        }
    }
}

impl<T, E> From<Expression<T, E>> for Result<T, E> {
    fn from(result: Expression<T, E>) -> Self {
        match result {
            Defined(defined) => Ok(defined),
            Undefined(undefined) => Err(undefined),
        }
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> FromResidual for Expression<T, E> {
    fn from_residual(error: Result<Infallible, E>) -> Self {
        match error {
            Err(error) => Undefined(error),
            _ => unreachable!(),
        }
    }
}

impl<T, P> Infinite for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Copy,
    Proxy<T, P>: Infinite,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    const INFINITY: Self = Defined(Infinite::INFINITY);
    const NEG_INFINITY: Self = Defined(Infinite::NEG_INFINITY);

    fn is_infinite(self) -> bool {
        match self {
            Defined(defined) => defined.is_infinite(),
            _ => false,
        }
    }

    fn is_finite(self) -> bool {
        match self {
            Defined(defined) => defined.is_finite(),
            _ => false,
        }
    }
}

impl<T, E> PartialEq for Expression<T, E>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Defined(ref left), Defined(ref right)) => left.eq(right),
            _ => false,
        }
    }
}

impl<T, E> PartialOrd for Expression<T, E>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (Defined(ref left), Defined(ref right)) => left.partial_cmp(right),
            _ => None,
        }
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> Try for Expression<T, E> {
    type Output = T;
    type Residual = Result<Infallible, E>;

    fn from_output(output: T) -> Self {
        Defined(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Defined(defined) => ControlFlow::Continue(defined),
            Undefined(undefined) => ControlFlow::Break(Err(undefined)),
        }
    }
}

impl<T, P> UnaryReal for ExpressionOf<Proxy<T, P>>
where
    ErrorOf<Proxy<T, P>>: Clone + UndefinedError,
    T: Float + Primitive,
    P: Constraint<Divergence = TryExpression>,
{
    const ZERO: Self = Defined(UnaryReal::ZERO);
    const ONE: Self = Defined(UnaryReal::ONE);
    const E: Self = Defined(UnaryReal::E);
    const PI: Self = Defined(UnaryReal::PI);
    const FRAC_1_PI: Self = Defined(UnaryReal::FRAC_1_PI);
    const FRAC_2_PI: Self = Defined(UnaryReal::FRAC_2_PI);
    const FRAC_2_SQRT_PI: Self = Defined(UnaryReal::FRAC_2_SQRT_PI);
    const FRAC_PI_2: Self = Defined(UnaryReal::FRAC_PI_2);
    const FRAC_PI_3: Self = Defined(UnaryReal::FRAC_PI_3);
    const FRAC_PI_4: Self = Defined(UnaryReal::FRAC_PI_4);
    const FRAC_PI_6: Self = Defined(UnaryReal::FRAC_PI_6);
    const FRAC_PI_8: Self = Defined(UnaryReal::FRAC_PI_8);
    const SQRT_2: Self = Defined(UnaryReal::SQRT_2);
    const FRAC_1_SQRT_2: Self = Defined(UnaryReal::FRAC_1_SQRT_2);
    const LN_2: Self = Defined(UnaryReal::LN_2);
    const LN_10: Self = Defined(UnaryReal::LN_10);
    const LOG2_E: Self = Defined(UnaryReal::LOG2_E);
    const LOG10_E: Self = Defined(UnaryReal::LOG10_E);

    fn is_zero(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_zero(),
            _ => false,
        }
    }

    fn is_one(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_one(),
            _ => false,
        }
    }

    fn is_positive(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_positive(),
            _ => false,
        }
    }

    fn is_negative(self) -> bool {
        match self.defined() {
            Some(defined) => defined.is_negative(),
            _ => false,
        }
    }

    #[cfg(feature = "std")]
    fn abs(self) -> Self {
        self.map(UnaryReal::abs)
    }

    #[cfg(feature = "std")]
    fn signum(self) -> Self {
        self.map(UnaryReal::signum)
    }

    fn floor(self) -> Self {
        self.map(UnaryReal::floor)
    }

    fn ceil(self) -> Self {
        self.map(UnaryReal::ceil)
    }

    fn round(self) -> Self {
        self.map(UnaryReal::round)
    }

    fn trunc(self) -> Self {
        self.map(UnaryReal::trunc)
    }

    fn fract(self) -> Self {
        self.map(UnaryReal::fract)
    }

    fn recip(self) -> Self::Superset {
        self.and_then(UnaryReal::recip)
    }

    #[cfg(feature = "std")]
    fn powi(self, n: i32) -> Self::Superset {
        self.and_then(|defined| UnaryReal::powi(defined, n))
    }

    #[cfg(feature = "std")]
    fn sqrt(self) -> Self::Superset {
        self.and_then(UnaryReal::sqrt)
    }

    #[cfg(feature = "std")]
    fn cbrt(self) -> Self {
        self.map(UnaryReal::cbrt)
    }

    #[cfg(feature = "std")]
    fn exp(self) -> Self::Superset {
        self.and_then(UnaryReal::exp)
    }

    #[cfg(feature = "std")]
    fn exp2(self) -> Self::Superset {
        self.and_then(UnaryReal::exp2)
    }

    #[cfg(feature = "std")]
    fn exp_m1(self) -> Self::Superset {
        self.and_then(UnaryReal::exp_m1)
    }

    #[cfg(feature = "std")]
    fn ln(self) -> Self::Superset {
        self.and_then(UnaryReal::ln)
    }

    #[cfg(feature = "std")]
    fn log2(self) -> Self::Superset {
        self.and_then(UnaryReal::log2)
    }

    #[cfg(feature = "std")]
    fn log10(self) -> Self::Superset {
        self.and_then(UnaryReal::log10)
    }

    #[cfg(feature = "std")]
    fn ln_1p(self) -> Self::Superset {
        self.and_then(UnaryReal::ln_1p)
    }

    #[cfg(feature = "std")]
    fn to_degrees(self) -> Self::Superset {
        self.and_then(UnaryReal::to_degrees)
    }

    #[cfg(feature = "std")]
    fn to_radians(self) -> Self {
        self.map(UnaryReal::to_radians)
    }

    #[cfg(feature = "std")]
    fn sin(self) -> Self {
        self.map(UnaryReal::sin)
    }

    #[cfg(feature = "std")]
    fn cos(self) -> Self {
        self.map(UnaryReal::cos)
    }

    #[cfg(feature = "std")]
    fn tan(self) -> Self::Superset {
        self.and_then(UnaryReal::tan)
    }

    #[cfg(feature = "std")]
    fn asin(self) -> Self::Superset {
        self.and_then(UnaryReal::asin)
    }

    #[cfg(feature = "std")]
    fn acos(self) -> Self::Superset {
        self.and_then(UnaryReal::acos)
    }

    #[cfg(feature = "std")]
    fn atan(self) -> Self {
        self.map(UnaryReal::atan)
    }

    #[cfg(feature = "std")]
    fn sin_cos(self) -> (Self, Self) {
        match self {
            Defined(defined) => {
                let (sin, cos) = defined.sin_cos();
                (Defined(sin), Defined(cos))
            }
            Undefined(undefined) => (Undefined(undefined.clone()), Undefined(undefined)),
        }
    }

    #[cfg(feature = "std")]
    fn sinh(self) -> Self {
        self.map(UnaryReal::sinh)
    }

    #[cfg(feature = "std")]
    fn cosh(self) -> Self {
        self.map(UnaryReal::cosh)
    }

    #[cfg(feature = "std")]
    fn tanh(self) -> Self {
        self.map(UnaryReal::tanh)
    }

    #[cfg(feature = "std")]
    fn asinh(self) -> Self::Superset {
        self.and_then(UnaryReal::asinh)
    }

    #[cfg(feature = "std")]
    fn acosh(self) -> Self::Superset {
        self.and_then(UnaryReal::acosh)
    }

    #[cfg(feature = "std")]
    fn atanh(self) -> Self::Superset {
        self.and_then(UnaryReal::atanh)
    }
}

macro_rules! impl_binary_operation {
    () => {
        with_binary_operations!(impl_binary_operation);
    };
    (operation => $trait:ident :: $method:ident) => {
        impl_binary_operation!(operation => $trait :: $method, |left, right| {
            left.zip_map(right, $trait::$method)
        });
    };
    (operation => $trait:ident :: $method:ident, |$left:ident, $right:ident| $f:block) => {
        macro_rules! impl_primitive_binary_operation {
            () => {
                with_primitives!(impl_primitive_binary_operation);
            };
            (primitive => $t:ty) => {
                impl<P> $trait<ExpressionOf<Proxy<$t, P>>> for $t
                where
                    P: Constraint<Divergence = TryExpression>,
                {
                    type Output = ExpressionOf<Proxy<$t, P>>;

                    fn $method(self, other: ExpressionOf<Proxy<$t, P>>) -> Self::Output {
                        let $left = expression!(Proxy::<_, P>::new(self));
                        let $right = expression!(other);
                        $f
                    }
                }
            };
        }
        impl_primitive_binary_operation!();

        impl<T, P> $trait<ExpressionOf<Self>> for Proxy<T, P>
        where
            T: Float + Primitive,
            P: Constraint<Divergence = TryExpression>,
        {
            type Output = ExpressionOf<Self>;

            fn $method(self, other: ExpressionOf<Self>) -> Self::Output {
                let $left = self;
                let $right = expression!(other);
                $f
            }
        }

        impl<T, P> $trait<Proxy<T, P>> for ExpressionOf<Proxy<T, P>>
        where
            T: Float + Primitive,
            P: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: Proxy<T, P>) -> Self::Output {
                let $left = expression!(self);
                let $right = other;
                $f
            }
        }

        impl<T, P> $trait<ExpressionOf<Proxy<T, P>>> for ExpressionOf<Proxy<T, P>>
        where
            T: Float + Primitive,
            P: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: Self) -> Self::Output {
                let $left = expression!(self);
                let $right = expression!(other);
                $f
            }
        }

        impl<T, P> $trait<T> for ExpressionOf<Proxy<T, P>>
        where
            T: Float + Primitive,
            P: Constraint<Divergence = TryExpression>,
        {
            type Output = Self;

            fn $method(self, other: T) -> Self::Output {
                let $left = expression!(self);
                let $right = expression!(Proxy::<_, P>::new(other));
                $f
            }
        }
    };
}
impl_binary_operation!();

macro_rules! impl_try_from {
    () => {
        with_primitives!(impl_try_from);
    };
    (primitive => $t:ty) => {
        impl<P> TryFrom<Expression<Proxy<$t, P>, P::Error>> for Proxy<$t, P>
        where
            P: Constraint,
        {
            type Error = P::Error;

            fn try_from(
                expression: Expression<Proxy<$t, P>, P::Error>,
            ) -> Result<Self, Self::Error> {
                match expression {
                    Defined(defined) => Ok(defined),
                    Undefined(undefined) => Err(undefined),
                }
            }
        }

        impl<P> TryFrom<Expression<Proxy<$t, P>, P::Error>> for $t
        where
            P: Constraint,
        {
            type Error = P::Error;

            fn try_from(
                expression: Expression<Proxy<$t, P>, P::Error>,
            ) -> Result<Self, Self::Error> {
                match expression {
                    Defined(defined) => Ok(defined.into()),
                    Undefined(undefined) => Err(undefined),
                }
            }
        }
    };
}
impl_try_from!();
