use std::env;
use std::path::PathBuf;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap();
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let crate_path = PathBuf::from(manifest_dir);

    match target_os.as_str() {
        "linux" => {
            let map_path = crate_path.join("exports.map");
            println!(
                "cargo:rustc-link-arg=-Wl,--version-script={}",
                map_path.display()
            );
            println!("cargo:rerun-if-changed=exports.map");
        }
        //"macos" => {
        //    let txt_path = crate_path.join("exports.txt");
        //    println!("cargo:rustc-link-arg=-Wl,-exported_symbols_list,{}", txt_path.display());
        //    println!("cargo:rerun-if-changed=exports.txt");
        //}
        _ => {
            // skip for now
        }
    }
}

/*
// --- МАГИЯ ДЛЯ MACOS ---

            // 1. Поднимаемся из OUT_DIR в папку профиля (debug или release), чтобы найти зависимости
            // OUT_DIR обычно: .../target/debug/build/slm_llama-XXXXX/out
            let target_profile_dir = out_dir
                .parent().unwrap() // slm_llama-XXXXX
                .parent().unwrap() // build
                .parent().unwrap(); // debug или release

            let build_dir = target_profile_dir.join("build");
            let mut llama_symbols = Vec::new();

            // 2. Ищем папку сборки llama-cpp-sys-2
            if let Ok(entries) = fs::read_dir(build_dir) {
                for entry in entries.flatten() {
                    let folder_name = entry.file_name().to_string_lossy().into_owned();
                    if folder_name.contains("llama-cpp-sys") {
                        let sys_out_dir = entry.path().join("out");

                        // 3. Сканируем файлы .a (libllama.a, libggml.a и т.д.)
                        if let Ok(files) = fs::read_dir(sys_out_dir) {
                            for file in files.flatten() {
                                if file.path().extension().map_or(false, |ext| ext == "a") {

                                    // 4. Запускаем системный `nm -g -j` (только глобальные символы, только имена)
                                    let output = Command::new("nm")
                                        .args(["-g", "-j", &file.path().to_string_lossy()])
                                        .output();

                                    if let Ok(out) = output {
                                        let stdout = String::from_utf8_lossy(&out.stdout);
                                        for line in stdout.lines() {
                                            // На macOS Си-символы в бинарнике всегда начинаются с нижнего подчеркивания
                                            if line.starts_with("_llama_") || line.starts_with("_ggml_") || line.starts_with("_gg_") {
                                                llama_symbols.push(line.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if llama_symbols.is_empty() {
                // Если nm ничего не нашел (например, на чистой системе без Xcode Command Line Tools),
                // выведи предупреждение, но не падай.
                println!("cargo:warning=Не удалось извлечь символы llama.cpp для скрытия на macOS");
            }

            // 5. Сохраняем полученный список в файл unexported.txt внутри OUT_DIR
            let unexported_txt_path = out_dir.join("unexported.txt");
            fs::write(&unexported_txt_path, llama_symbols.join("\n")).unwrap();

            // 6. Передаем этот файл Мак-линкеру как список Скрываемых символов
            println!("cargo:rustc-link-arg=-Wl,-unexported_symbols_list,{}", unexported_txt_path.display());
        }
 */
