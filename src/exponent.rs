use crate::data;
use crate::render;

pub fn analyze() {
    let param_a: f32 = 0.35;
    let param_b: f32 = 0.0004;

    let mut train = Vec::with_capacity(data::VALUES.len());

    let mut prev = ForecastResult {
        l: data::VALUES[0],
        b: 1.0,
        y: data::VALUES[0],
    };
    let mut h: f32 = 1.0;
    train.push(prev.y);

    for i in 1..data::VALUES.len() {
        let predict = forecast_m_n(data::VALUES[i], prev.l, prev.b, param_a, param_b, h);
        train.push(predict.y);
        prev = predict;
        h += 1.0;
    }

    let mae = data::mae(&train, &data::VALUES);
    let rmse = data::rmse(data::mse(&train, &data::VALUES));

    for i in 0..data::SEASON * 3 {
        let predict = forecast_m_n(prev.y, prev.l, prev.b, param_a, param_b, h);
        train.push(predict.y);
        prev = predict;
        h += 1.0;
    }

    println!("\nError analysis for `exponential M-N` method");
    println!("MAE: {mae}");
    println!("RMSE: {rmse}");
    render::render(&train, false, Some("lab_2"), "exponent_m_n");
}

struct ForecastResult {
    l: f32,
    b: f32,
    y: f32,
}

fn forecast_m_n(
    real_y: f32,
    prev_l: f32,
    prev_b: f32,
    param_a: f32,
    param_b: f32,
    h: f32,
) -> ForecastResult {
    let l = param_a * real_y + (1.0 - param_a) * prev_l * prev_b;
    let b = param_b * (l / prev_l) + (1.0 - param_b) * prev_b;
    let y = l * b.powf(h);
    ForecastResult { l, b, y }
}
