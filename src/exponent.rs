use crate::data;
use crate::render;

pub fn analyze() {
    let mut train = [0.0_f32; data::SEASON * 4];
    let mut predicted = [0.0_f32; data::SEASON];
    let test = &data::VALUES[data::TEST_RANGE];

    const PARAM_A_STEP: f32 = 0.001; // 1000 iter
    const PARAM_B_STEP: f32 = 0.001; // 1000 iter

    let mut param_a: f32 = PARAM_A_STEP;
    let mut param_b: f32 = PARAM_B_STEP;

    let mut iterations = 0;
    let mut min_rmse: f32 = f32::MAX;
    let mut best_mae: f32 = 0.0;
    let mut best_param_a: f32 = param_a;
    let mut best_param_b: f32 = param_b;
    let mut best_train = [0.0_f32; data::SEASON * 4];
    let mut best_predicted = [0.0_f32; data::SEASON];

    while param_a < 1.0 {
        while param_b < 1.0 {
            let prev = smooth_real_values(train.as_mut_slice(), param_a, param_b);
            let eval =
                predict_test_next_values(test, predicted.as_mut_slice(), prev, param_a, param_b);

            iterations += 1;
            if eval.rmse.is_finite() && eval.rmse < min_rmse {
                min_rmse = eval.rmse;
                best_mae = eval.mae;
                best_param_a = param_a;
                best_param_b = param_b;
                best_train = train;
                best_predicted = predicted;
            }
            if param_b < 1.0 {
                param_b += PARAM_B_STEP;
            }
        }

        param_b = PARAM_B_STEP;
        if param_a < 1.0 {
            param_a += PARAM_A_STEP;
        }
    }

    // best training result
    {
        println!("\nError analysis for `exponential M-N v2` method");
        println!("BEST MAE: {}", best_mae);
        println!("BEST RMSE: {}", min_rmse);
        println!("BEST PARAMS: A `{best_param_a}`, B `{best_param_b}`");
        println!("TRAIN ITERATIONS: {iterations}");

        let mut joined = best_train.to_vec();
        joined.extend(best_predicted);
        render::render(&joined, false, Some("lab_2"), "exponent_m_n_training");
    }

    // future forecast
    {
        let mut values = [0.0_f32; data::SEASON * 5];
        let mut predicted = [0.0_f32; data::SEASON];

        let prev = smooth_real_values(values.as_mut_slice(), best_param_a, best_param_b);
        predict_next_values(predicted.as_mut_slice(), prev, best_param_a, best_param_b);

        let mut joined = values.to_vec();
        joined.extend(predicted);
        render::render(&joined, false, Some("lab_2"), "exponent_m_n_forecast");
    }
}

fn smooth_real_values(target: &mut [f32], param_a: f32, param_b: f32) -> ForecastResult {
    let mut prev = ForecastResult {
        l: data::VALUES[0],
        b: 1.0,
        y: data::VALUES[0],
    };
    target[0] = prev.y;

    for i in 1..target.len() {
        let predict = forecast_m_n(data::VALUES[i], prev.l, prev.b, param_a, param_b, 1.0);
        target[i] = predict.y;
        prev = predict;
    }
    return prev;
}

fn predict_test_next_values(
    test: &[f32],
    predicted: &mut [f32],
    mut prev: ForecastResult,
    param_a: f32,
    param_b: f32,
) -> ForecastEval {
    let mut h = 1.0;
    for i in 0..predicted.len() {
        let predict = forecast_m_n(prev.y, prev.l, prev.b, param_a, param_b, h);
        predicted[i] = predict.y;
        prev = predict;
        h += 1.0;
    }
    return ForecastEval {
        mae: data::mae(predicted, test),
        rmse: data::rmse(data::mse(predicted, test)),
    };
}

fn predict_next_values(
    predicted: &mut [f32],
    mut prev: ForecastResult,
    param_a: f32,
    param_b: f32,
) {
    let mut h = 1.0;
    for i in 0..predicted.len() {
        let predict = forecast_m_n(prev.y, prev.l, prev.b, param_a, param_b, h);
        predicted[i] = predict.y;
        prev = predict;
        h += 1.0;
    }
}

struct ForecastResult {
    l: f32,
    b: f32,
    y: f32,
}

struct ForecastEval {
    mae: f32,
    rmse: f32,
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
