use core::convert::Infallible;
use core::fmt::Debug;
#[cfg(all(nightly, feature = "unstable"))]
use core::ops::{ControlFlow, FromResidual, Try};

use crate::constraint::ExpectConstrained as _;
use crate::proxy::{ClosedProxy, ErrorOf};

pub use ConstraintResult::Err as FloatErr;
pub use ConstraintResult::Ok as FloatOk;

pub trait ErrorMode {
    type Branch<T, E>;

    fn from_output<T, E>(output: T) -> Self::Branch<T, E>;

    fn from_residual<T, E>(residual: E) -> Self::Branch<T, E>
    where
        E: Debug;
}

impl ErrorMode for Infallible {
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

pub trait NonResidual<T>: ErrorMode<Branch<T, ErrorOf<T>> = T>
where
    T: ClosedProxy,
{
}

impl<T, M> NonResidual<T> for M
where
    T: ClosedProxy,
    M: ErrorMode<Branch<T, ErrorOf<T>> = T>,
{
}

pub trait ResidualBranch {}

impl<T> ResidualBranch for Option<T> {}

impl<T, E> ResidualBranch for Result<T, E> {}

pub enum Assert {}

impl ErrorMode for Assert {
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

pub enum TryOption {}

impl ErrorMode for TryOption {
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

impl ErrorMode for TryResult {
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

// --- --- ---

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
impl<T, E> Try for ConstraintResult<T, E> {
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
