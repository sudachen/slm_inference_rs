use std::marker::PhantomData;
use std::sync::Arc;

pub trait Free<T> {
    unsafe fn free(ptr: *mut T);
}

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
    pub fn new(ptr: *mut T) -> Self {
        Self {
            ptr,
            _marker: PhantomData,
        }
    }

    pub fn get_const_ptr(&self) -> *const T {
        self.ptr
    }
    pub fn get_ptr(&self) -> *mut T {
        self.ptr
    }
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

#[derive(Clone)]
pub struct SharedPtr<T, F: Free<T>> {
    inner: Arc<UniquePtr<T, F>>,
}

pub trait ThreadSafe {}
unsafe impl<T, F: Free<T> + ThreadSafe> Send for SharedPtr<T, F> {}
unsafe impl<T, F: Free<T> + ThreadSafe> Sync for SharedPtr<T, F> {}

impl<T, F: Free<T>> SharedPtr<T, F> {
    pub fn new(ptr: *mut T) -> Self {
        Self {
            inner: Arc::new(UniquePtr::new(ptr)),
        }
    }

    pub fn get_const_ptr(&self) -> *const T {
        self.inner.ptr
    }
    pub fn get_ptr(&self) -> *mut T {
        self.inner.ptr
    }
    pub fn is_null(&self) -> bool {
        self.inner.ptr.is_null()
    }
}
