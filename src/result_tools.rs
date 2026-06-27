use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::{ThetaJobResult, ThetaTaskResult};

#[derive(Debug, Clone, PartialEq)]
pub struct ResultDataFrame {
    pub columns: Vec<String>,
    pub rows: Vec<BTreeMap<String, String>>,
}

impl ResultDataFrame {
    pub fn write_csv(&self, path: impl AsRef<Path>) -> Result<PathBuf, String> {
        self.write_delimited(path, ',')
    }

    pub fn write_tsv(&self, path: impl AsRef<Path>) -> Result<PathBuf, String> {
        self.write_delimited(path, '\t')
    }

    pub fn write_delimited(
        &self,
        path: impl AsRef<Path>,
        delimiter: char,
    ) -> Result<PathBuf, String> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|err| err.to_string())?;
        }
        let mut file = File::create(path).map_err(|err| err.to_string())?;
        write_delimited_record(&mut file, &self.columns, delimiter)?;
        for row in &self.rows {
            let fields = self
                .columns
                .iter()
                .map(|column| row.get(column).cloned().unwrap_or_default())
                .collect::<Vec<_>>();
            write_delimited_record(&mut file, &fields, delimiter)?;
        }
        Ok(path.to_path_buf())
    }
}

pub fn dataframe(result_json: impl AsRef<Path>) -> Result<ResultDataFrame, String> {
    let file = File::open(&result_json).map_err(|err| err.to_string())?;
    let result: ThetaJobResult =
        serde_json::from_reader(BufReader::new(file)).map_err(|err| err.to_string())?;
    Ok(dataframe_from_result(&result))
}

pub fn dataframe_from_result(result: &ThetaJobResult) -> ResultDataFrame {
    let mut observable_names = BTreeSet::new();
    for task in &result.tasks {
        observable_names.extend(task.observables.keys().cloned());
    }

    let mut columns = vec![
        "job_name".to_string(),
        "rank".to_string(),
        "world_size".to_string(),
        "task_index".to_string(),
        "task_name".to_string(),
        "L".to_string(),
        "Lx".to_string(),
        "Ly".to_string(),
        "Lz".to_string(),
        "T".to_string(),
        "Jxy".to_string(),
        "Jz".to_string(),
        "DeltaJz".to_string(),
        "sample".to_string(),
        "seed".to_string(),
        "disorder_seed".to_string(),
        "sweeps".to_string(),
        "thermalization".to_string(),
        "binsize".to_string(),
        "measurements".to_string(),
        "acceptance".to_string(),
    ];
    for name in &observable_names {
        columns.push(name.clone());
        columns.push(format!("{name}_error"));
        columns.push(format!("{name}_measurement"));
    }

    let rows = result
        .tasks
        .iter()
        .map(|task| dataframe_row(result, task))
        .collect();

    ResultDataFrame { columns, rows }
}

pub fn write_dataframe(
    result_json: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<PathBuf, String> {
    let frame = dataframe(result_json)?;
    match output_path
        .as_ref()
        .extension()
        .and_then(|extension| extension.to_str())
    {
        Some("csv") => frame.write_csv(output_path),
        _ => frame.write_tsv(output_path),
    }
}

pub fn write_gnuplot_script(
    table_path: impl AsRef<Path>,
    script_path: impl AsRef<Path>,
    image_path: impl AsRef<Path>,
    observable: &str,
) -> Result<PathBuf, String> {
    let table_path = table_path.as_ref();
    let script_path = script_path.as_ref();
    let image_path = image_path.as_ref();
    if let Some(parent) = script_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }
    if let Some(parent) = image_path.parent() {
        fs::create_dir_all(parent).map_err(|err| err.to_string())?;
    }

    let escaped_table = gnuplot_quote(&table_path.to_string_lossy());
    let escaped_image = gnuplot_quote(&image_path.to_string_lossy());
    let escaped_observable = gnuplot_quote(observable);
    let script = format!(
        "set terminal pngcairo size 1000,700\n\
set output '{escaped_image}'\n\
set datafile separator '\\t'\n\
set key title 'L'\n\
set xlabel 'Temperature'\n\
set ylabel '{escaped_observable}'\n\
set title '{escaped_observable} vs Temperature'\n\
set grid\n\
plot '{escaped_table}' using 'T':'{escaped_observable}':'Lx' with points pointtype 7 palette title columnheader\n"
    );
    fs::write(script_path, script).map_err(|err| err.to_string())?;
    Ok(script_path.to_path_buf())
}

pub fn plot_with_gnuplot(script_path: impl AsRef<Path>) -> Result<(), String> {
    let output = Command::new("gnuplot")
        .arg(script_path.as_ref())
        .output()
        .map_err(|err| format!("failed to run gnuplot: {err}"))?;
    if output.status.success() {
        Ok(())
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        Err(format!("gnuplot failed: {stderr}"))
    }
}

fn dataframe_row(
    result: &ThetaJobResult,
    task_result: &ThetaTaskResult,
) -> BTreeMap<String, String> {
    let task = &task_result.task;
    let mut row = BTreeMap::new();
    row.insert("job_name".to_string(), result.job_name.clone());
    row.insert("rank".to_string(), result.rank.to_string());
    row.insert("world_size".to_string(), result.world_size.to_string());
    row.insert("task_index".to_string(), task_result.task_index.to_string());
    row.insert("task_name".to_string(), task.name.clone());
    row.insert("L".to_string(), task.l.to_string());
    row.insert("Lx".to_string(), task.l_x.to_string());
    row.insert("Ly".to_string(), task.l_y.to_string());
    row.insert("Lz".to_string(), task.l_z.to_string());
    row.insert("T".to_string(), format_float(task.temperature));
    row.insert("Jxy".to_string(), format_float(task.j_xy));
    row.insert("Jz".to_string(), format_float(task.j_z_mean));
    row.insert("DeltaJz".to_string(), format_float(task.delta_j_z));
    row.insert("sample".to_string(), task.sample.to_string());
    row.insert("seed".to_string(), task.seed.to_string());
    row.insert("disorder_seed".to_string(), task.disorder_seed.to_string());
    row.insert("sweeps".to_string(), task.sweeps.to_string());
    row.insert(
        "thermalization".to_string(),
        task.thermalization.to_string(),
    );
    row.insert("binsize".to_string(), task.binsize.to_string());
    row.insert(
        "measurements".to_string(),
        task_result.measurements.to_string(),
    );
    row.insert(
        "acceptance".to_string(),
        format_float(task_result.acceptance),
    );

    for (name, estimate) in &task_result.observables {
        row.insert(name.clone(), format_float(estimate.mean));
        row.insert(format!("{name}_error"), format_float(estimate.error));
        row.insert(
            format!("{name}_measurement"),
            format!(
                "{} +/- {}",
                format_float(estimate.mean),
                format_float(estimate.error)
            ),
        );
    }
    row
}

fn write_delimited_record(
    writer: &mut impl Write,
    fields: &[String],
    delimiter: char,
) -> Result<(), String> {
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            write!(writer, "{delimiter}").map_err(|err| err.to_string())?;
        }
        write!(writer, "{}", escape_delimited_field(field, delimiter))
            .map_err(|err| err.to_string())?;
    }
    writeln!(writer).map_err(|err| err.to_string())
}

fn escape_delimited_field(field: &str, delimiter: char) -> String {
    if field.contains(delimiter)
        || field.contains('\n')
        || field.contains('\r')
        || field.contains('"')
    {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

fn format_float(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.16}")
    } else if value.is_nan() {
        "NaN".to_string()
    } else if value.is_sign_negative() {
        "-Inf".to_string()
    } else {
        "Inf".to_string()
    }
}

fn gnuplot_quote(value: &str) -> String {
    value.replace('\\', "\\\\").replace('\'', "\\'")
}
