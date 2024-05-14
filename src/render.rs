use crate::data;
use plotters::{prelude::*, style::full_palette as palette};

const CAPTION: &str = "Дані про сумарні місячні продажі товарів в магазинах";
const Y_NAME: &str = "MONTH";
const X_NAME: &str = "SALES";

pub fn render(data: &[f32], mix_color: bool, dir: Option<&str>, name: &str) {
    let mut path = std::env::current_dir().expect("cwd");
    path.push("forecast_graphs");
    if !path.exists() {
        std::fs::create_dir(&path).expect("dir created");
    }
    if let Some(dir) = dir {
        path.push(dir);
        if !path.exists() {
            std::fs::create_dir(&path).expect("dir created");
        }
    }
    path.push(format!("{}.png", name));

    let root = BitMapBackend::new(&path, (1600, 900)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .x_label_area_size(50)
        .y_label_area_size(50)
        .margin(10)
        .caption(CAPTION, ("sans-serif", 36))
        .build_cartesian_2d((0..data.len() as u32).into_segmented(), 0..3000 as u32)
        .unwrap();

    chart
        .configure_mesh()
        .disable_x_mesh()
        .bold_line_style(WHITE.mix(0.3))
        .y_desc(X_NAME)
        .x_desc(Y_NAME)
        .axis_desc_style(("sans-serif", 15))
        .draw()
        .unwrap();

    let colors = if mix_color {
        vec![
            &palette::ORANGE,
            &GREEN,
            &MAGENTA,
            &BLUE,
            &palette::PURPLE,
            &RED,
            &RED,
            &RED,
        ]
    } else {
        vec![&RED]
    };

    for (i, chunk) in data.chunks(data::SEASON).enumerate() {
        let color_index = i % colors.len();
        let color = colors[color_index].mix(0.5).filled();

        chart
            .draw_series(Histogram::vertical(&chart).style(color).data(
                chunk.iter().enumerate().map(|(idx, &count)| {
                    (i as u32 * data::SEASON as u32 + idx as u32, count as u32)
                }),
            ))
            .unwrap();
    }

    root.present().expect("unable to write output graph");
    println!("graph has been saved to `{}`", path.to_string_lossy());
}
