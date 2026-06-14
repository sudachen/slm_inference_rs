use std::ffi::CStr;
use std::os::raw::{c_char, c_uint, c_void};
use std::sync::atomic::{AtomicBool, Ordering};
use tracing::{error, trace, warn};

static LLAMA_BACKEND_INITIALIZED: AtomicBool = AtomicBool::new(false);

pub struct Backend;

#[inline(never)]
pub fn init() -> Backend {
    if LLAMA_BACKEND_INITIALIZED
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_ok()
    {
        unsafe {
            slm_ikllama_sys::llama_backend_init();
            slm_ikllama_sys::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
        }
    }
    Backend {}
}


#[inline(never)]
unsafe extern "C" fn llama_log_callback(
    level: c_uint,
    text: *const c_char,
    _user_data: *mut c_void,
) {
    if text.is_null() {
        return;
    }

    let c_str = unsafe { CStr::from_ptr(text) };
    let msg = c_str.to_string_lossy().trim_end().to_string();

    // Маппим уровни логов GGML на tracing
    match level {
        slm_ikllama_sys::GGML_LOG_LEVEL_ERROR => error!("ik_llama.cpp: {}", msg),
        slm_ikllama_sys::GGML_LOG_LEVEL_WARN => warn!("ik_llama.cpp: {}", msg),
        _ => trace!("ik_llama.cpp: {}", msg),
    }
}
