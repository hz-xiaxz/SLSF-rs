pub fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

pub fn sample_std(x: &[f64]) -> f64 {
    if x.len() <= 1 {
        return 0.0;
    }
    let avg = mean(x);
    let variance = x.iter().map(|v| (v - avg).powi(2)).sum::<f64>() / (x.len() - 1) as f64;
    variance.sqrt()
}

pub fn autocorrelation_function(x: &[f64], maxlag: Option<usize>) -> Vec<f64> {
    let n = x.len();
    if n <= 1 {
        return Vec::new();
    }
    let maxlag = maxlag.unwrap_or_else(|| (n - 1).min(200));
    let avg = mean(x);
    let centered: Vec<f64> = x.iter().map(|v| v - avg).collect();
    let c0 = centered.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if c0 <= 0.0 {
        return vec![1.0; maxlag + 1];
    }
    let mut acf = vec![0.0; maxlag + 1];
    acf[0] = 1.0;
    for lag in 1..=maxlag {
        let dot = centered[..n - lag]
            .iter()
            .zip(&centered[lag..])
            .map(|(a, b)| a * b)
            .sum::<f64>();
        acf[lag] = dot / ((n - lag) as f64 * c0);
    }
    acf
}

pub fn integrated_autocorrelation_time(x: &[f64], maxlag: Option<usize>) -> f64 {
    let acf = autocorrelation_function(x, maxlag);
    if acf.is_empty() {
        return 0.0;
    }
    let mut tau = 0.5;
    for &value in acf.iter().skip(1) {
        if value <= 0.0 {
            break;
        }
        tau += value;
    }
    tau
}

pub fn blocking_stderr(x: &[f64]) -> f64 {
    if x.len() <= 1 {
        return 0.0;
    }
    let mut data = x.to_vec();
    let mut stderr = sample_std(&data) / (data.len() as f64).sqrt();
    while data.len() >= 16 {
        let even_length = data.len() - (data.len() % 2);
        data = (0..even_length)
            .step_by(2)
            .map(|i| (data[i] + data[i + 1]) / 2.0)
            .collect();
        let candidate = sample_std(&data) / (data.len() as f64).sqrt();
        stderr = stderr.max(candidate);
    }
    stderr
}
