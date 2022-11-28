use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{ControlFlow, FromResidual, Try as TryOperation};

use crate::constraint::{Constraint, Error, ExpectConstrained as _};
use crate::proxy::ClosedProxy;

pub use ConstraintResult::Err as FloatErr;
pub use ConstraintResult::Ok as FloatOk;

pub type PrimitiveBranch<T> = <<T as ClosedProxy>::ErrorMode as ErrorMode>::Branch<
    <T as ClosedProxy>::Primitive,
    <<T as ClosedProxy>::Constraint as Constraint>::Error,
>;
pub type ProxyBranch<T> = <<T as ClosedProxy>::ErrorMode as ErrorMode>::Branch<
    T,
    <<T as ClosedProxy>::Constraint as Constraint>::Error,
>;

//#[cfg(not(all(nightly, feature = "unstable")))]
//pub type TryResult<T, E> = Result<T, E>;
//#[cfg(all(nightly, feature = "unstable"))]
//pub type TryResult<T, E> = ConstraintResult<T, E>;

pub trait ErrorMode {
    type Branch<T, E>;

    fn branch<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug;
}

impl ErrorMode for Infallible {
    type Branch<T, E> = T;

    fn branch<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        match result {
            Ok(inner) => inner,
            _ => unreachable!(),
        }
    }
}

pub trait NonResidual<T>: ErrorMode<Branch<T, Error<T>> = T>
where
    T: ClosedProxy,
{
}

impl<T, M> NonResidual<T> for M
where
    T: ClosedProxy,
    M: ErrorMode<Branch<T, Error<T>> = T>,
{
}

pub enum Assert {}

impl ErrorMode for Assert {
    type Branch<T, E> = T;

    fn branch<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        result.expect_constrained()
    }
}

pub enum TryOption {}

impl ErrorMode for TryOption {
    type Branch<T, E> = Option<T>;

    fn branch<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        result.ok()
    }
}

pub enum TryResult {}

impl ErrorMode for TryResult {
    type Branch<T, E> = Result<T, E>;

    fn branch<T, E>(result: Result<T, E>) -> Self::Branch<T, E>
    where
        E: Debug,
    {
        result
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ConstraintResult<T, E> {
    Ok(T),
    Err(E),
}

impl<T, E> From<Result<T, E>> for ConstraintResult<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(output) => FloatOk(output),
            Err(error) => FloatErr(error),
        }
    }
}

impl<T, E> From<ConstraintResult<T, E>> for Result<T, E> {
    fn from(result: ConstraintResult<T, E>) -> Self {
        match result {
            FloatOk(output) => Ok(output),
            FloatErr(error) => Err(error),
        }
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> FromResidual for ConstraintResult<T, E> {
    fn from_residual(error: Result<Infallible, E>) -> Self {
        match error {
            Err(error) => FloatErr(error),
            _ => unreachable!(),
        }
    }
}

#[cfg(all(nightly, feature = "unstable"))]
impl<T, E> TryOperation for ConstraintResult<T, E> {
    type Output = T;
    type Residual = Result<Infallible, E>;

    fn from_output(output: T) -> Self {
        FloatOk(output)
    }

    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            FloatOk(output) => ControlFlow::Continue(output),
            FloatErr(error) => ControlFlow::Break(Err(error)),
        }
    }
}
