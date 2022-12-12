use core::cmp::Ordering;
use core::convert::{Infallible, TryFrom};
use core::fmt::Debug;
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{ControlFlow, FromResidual, Try};

use crate::constraint::{Constraint, ExpectConstrained as _};
use crate::proxy::{ClosedProxy, ErrorOf, Proxy};
use crate::{Float, Primitive};

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

macro_rules! impl_try_from {
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
impl_try_from!(primitive => f32);
impl_try_from!(primitive => f64);
