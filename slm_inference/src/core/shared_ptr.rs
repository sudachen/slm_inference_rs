use std::marker::PhantomData;
use std::sync::Arc;

/// Trait for types that know how to free a raw pointer to `T`.
///
/// Implementors pair with [`UniquePtr`] / [`SharedPtr`] to provide RAII ownership
/// semantics over C-allocated objects obtained through FFI.
pub trait Free<T> {
    /// Release the memory pointed to by `ptr`.
    ///
    /// # Safety
    /// `ptr` must have been allocated by the corresponding C allocator and must not
    /// be used after this call.
    unsafe fn free(ptr: *mut T);
}

/// Single-owner RAII wrapper around a raw C pointer.
///
/// Calls `F::free` on the pointer when dropped.  Not `Clone`; use [`SharedPtr`]
/// when shared ownership is required.
pub struct UniquePtr<T, F: Free<T>> {
    ptr: *mut T,
    _marker: PhantomData<F>,
}

unsafe impl<T, F: Free<T>> Send for UniquePtr<T, F> {}

impl<T, F: Free<T>> Drop for UniquePtr<T, F> {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            unsafe {
                F::free(self.ptr);
            }
        }
    }
}

impl<T, F: Free<T>> UniquePtr<T, F> {
    /// Wrap a raw pointer.  Takes ownership; the pointer will be freed on drop.
    pub fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    /// Returns a read-only raw pointer to the underlying value.
    pub fn get_const_ptr(&self) -> *const T {
        self.ptr
    }
    /// Returns a mutable raw pointer to the underlying value.
    pub fn get_ptr(&self) -> *mut T {
        self.ptr
    }
    /// Returns `true` if the wrapped pointer is null.
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

/// Reference-counted shared wrapper around a raw C pointer.
///
/// The underlying pointer is freed (via `F::free`) when the last `SharedPtr`
/// clone is dropped.
#[derive(Clone)]
pub struct SharedPtr<T, F: Free<T>> {
    inner: Arc<UniquePtr<T, F>>,
}

/// Marker trait asserting that the pointed-to type and its `Free` implementation
/// are safe to send across threads and share between threads concurrently.
pub trait ThreadSafe {}
unsafe impl<T, F: Free<T> + ThreadSafe> Send for SharedPtr<T, F> {}
unsafe impl<T, F: Free<T> + ThreadSafe> Sync for SharedPtr<T, F> {}

impl<T, F: Free<T>> SharedPtr<T, F> {
    /// Wrap a raw pointer in a new `SharedPtr` with an initial reference count of 1.
    pub fn new(ptr: *mut T) -> Self {
        Self {
            inner: Arc::new(UniquePtr::new(ptr)),
        }
    }

    /// Returns a read-only raw pointer to the underlying value.
    pub fn get_const_ptr(&self) -> *const T {
        self.inner.ptr
    }
    /// Returns a mutable raw pointer to the underlying value.
    pub fn get_ptr(&self) -> *mut T {
        self.inner.ptr
    }
    /// Returns `true` if the wrapped pointer is null.
    pub fn is_null(&self) -> bool {
        self.inner.ptr.is_null()
    }
}
