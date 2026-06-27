fn main() {
    match slsf::run_theta_job_command(std::env::args_os()) {
        Ok(message) => println!("{message}"),
        Err(err) => {
            eprintln!("theta job failed: {err}");
            std::process::exit(1);
        }
    }
}
