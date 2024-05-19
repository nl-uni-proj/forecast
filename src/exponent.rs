pub fn analyze() {
    m_n::analyze();
    md_m::analyze();
}

mod m_n {
    use crate::data;
    use crate::render;

    struct ForecastEval {
        mae: f32,
        rmse: f32,
    }

    struct ForecastResult {
        l: f32,
        b: f32,
        y: f32,
    }

    fn forecast_m_n(
        y: f32,
        prev_l: f32,
        prev_b: f32,
        param_a: f32,
        param_b: f32,
        h: f32,
    ) -> ForecastResult {
        let l = param_a * y + (1.0 - param_a) * prev_l * prev_b;
        let b = param_b * (l / prev_l) + (1.0 - param_b) * prev_b;
        let y = l * b.powf(h);
        ForecastResult { l, b, y }
    }

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
                let eval = predict_test_next_values(
                    test,
                    predicted.as_mut_slice(),
                    prev,
                    param_a,
                    param_b,
                );

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
            println!("\nError analysis for `exponential M-N` method");
            println!("BEST MAE: {}", best_mae);
            println!("BEST RMSE: {}", min_rmse);
            println!("BEST PARAMS: A `{best_param_a}`, B `{best_param_b}`");
            println!("TRAIN ITERATIONS: {iterations}\n");

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
        prev: ForecastResult,
        param_a: f32,
        param_b: f32,
    ) -> ForecastEval {
        let mut h = 1.0;
        for i in 0..predicted.len() {
            let predict = forecast_m_n(prev.y, prev.l, prev.b, param_a, param_b, h);
            predicted[i] = predict.y;
            h += 1.0;
        }
        return ForecastEval {
            mae: data::mae(predicted, test),
            rmse: data::rmse(data::mse(predicted, test)),
        };
    }

    fn predict_next_values(
        predicted: &mut [f32],
        prev: ForecastResult,
        param_a: f32,
        param_b: f32,
    ) {
        let mut h = 1.0;
        for i in 0..predicted.len() {
            let predict = forecast_m_n(prev.y, prev.l, prev.b, param_a, param_b, h);
            predicted[i] = predict.y;
            h += 1.0;
        }
    }
}

mod md_m {
    use crate::data;
    use crate::render;

    struct ForecastEval {
        mae: f32,
        rmse: f32,
    }

    struct ForecastResult {
        l: f32,
        b: f32,
        s: f32,
        y: f32,
    }

    fn forecast_md_m(
        y: f32,
        prev_l: f32,
        prev_b: f32,
        prev_season_s: f32,
        param_a: f32,
        param_b: f32,
        param_y: f32,
        fi_damp: f32,
        h: f32,
    ) -> ForecastResult {
        let hm_mod = (h - 1.0) % data::SEASON_F + 1.0;
        let l = param_a * (y / prev_season_s) + (1.0 - param_a) * prev_l * prev_b.powf(fi_damp);
        let b = param_b * (l / prev_l) + (1.0 - param_b) * prev_b.powf(fi_damp);
        let s = param_y * (y / (prev_l * prev_b.powf(fi_damp))) + (1.0 - param_y) * prev_season_s;
        let y = l * b.powf(fi_damp * h) * prev_season_s + hm_mod;
        ForecastResult { l, b, s, y }
    }

    const FI_DAMP: f32 = 1.1;

    pub fn analyze() {
        let mut train = [0.0_f32; data::SEASON * 4];
        let mut predicted = [0.0_f32; data::SEASON];
        let mut prev_season_s = [1.0_f32; data::SEASON];
        let test = &data::VALUES[data::TEST_RANGE];

        const PARAM_A_STEP: f32 = 0.01; // 100 iter
        const PARAM_B_STEP: f32 = 0.01; // 100 iter
        const PARAM_Y_STEP: f32 = 0.01; // 100 iter 1m total

        let mut param_a: f32 = PARAM_A_STEP;
        let mut param_b: f32 = PARAM_B_STEP;
        let mut param_y: f32 = PARAM_Y_STEP;

        let mut iterations = 0;
        let mut min_rmse: f32 = f32::MAX;
        let mut best_mae: f32 = 0.0;
        let mut best_param_a: f32 = param_a;
        let mut best_param_b: f32 = param_b;
        let mut best_param_y: f32 = param_y;
        let mut best_train = [0.0_f32; data::SEASON * 4];
        let mut best_predicted = [0.0_f32; data::SEASON];

        for a_m in 1..100 {
            for b_m in 1..100 {
                for y_m in 1..100 {
                    param_a = PARAM_A_STEP * a_m as f32;
                    param_b = PARAM_B_STEP * b_m as f32;
                    param_y = PARAM_Y_STEP * y_m as f32;

                    let prev = smooth_real_values(
                        train.as_mut_slice(),
                        prev_season_s.as_mut_slice(),
                        param_a,
                        param_b,
                        param_y,
                    );
                    let eval = predict_test_next_values(
                        test,
                        predicted.as_mut_slice(),
                        prev_season_s.as_slice(),
                        prev,
                        param_a,
                        param_b,
                        param_y,
                    );

                    iterations += 1;
                    if eval.rmse.is_finite() && eval.rmse < min_rmse {
                        min_rmse = eval.rmse;
                        best_mae = eval.mae;
                        best_param_a = param_a;
                        best_param_b = param_b;
                        best_param_y = param_y;
                        best_train = train;
                        best_predicted = predicted;
                    }
                }
            }
        }

        // best training result
        {
            println!("\nError analysis for `exponential Md-M` method");
            println!("BEST MAE: {}", best_mae);
            println!("BEST RMSE: {}", min_rmse);
            println!("BEST PARAMS: A `{best_param_a}`, B `{best_param_b}`, Y `{best_param_y}`");
            println!("TRAIN ITERATIONS: {iterations}\n");

            let mut joined = best_train.to_vec();
            joined.extend(best_predicted);
            render::render(&joined, false, Some("lab_2"), "exponent_md_m_training");
        }

        // future forecast
        {
            let mut values = [0.0_f32; data::SEASON * 5];
            let mut predicted = [0.0_f32; data::SEASON];
            let mut prev_season_s = [1.0_f32; data::SEASON];

            let prev = smooth_real_values(
                values.as_mut_slice(),
                prev_season_s.as_mut_slice(),
                best_param_a,
                best_param_b,
                best_param_y,
            );
            predict_next_values(
                predicted.as_mut_slice(),
                prev_season_s.as_slice(),
                prev,
                best_param_a,
                best_param_b,
                best_param_y,
            );

            let mut joined = values.to_vec();
            joined.extend(predicted);
            render::render(&joined, false, Some("lab_2"), "exponent_md_m_forecast");
        }
    }

    fn smooth_real_values(
        target: &mut [f32],
        prev_season_s: &mut [f32],
        param_a: f32,
        param_b: f32,
        param_y: f32,
    ) -> ForecastResult {
        let mut prev = ForecastResult {
            l: data::VALUES[0],
            b: 1.0,
            s: 1.0,
            y: data::VALUES[0],
        };
        target[0] = prev.y;

        let mut prev_season_s_buffer = [1.0_f32; data::SEASON];
        let mut curr_season_s_buffer = [1.0_f32; data::SEASON];

        for i in 1..target.len() {
            if i % 12 == 0 {
                prev_season_s_buffer = curr_season_s_buffer;
            }

            let seasonal_idx = i % data::SEASON;
            let s_value = prev_season_s_buffer[seasonal_idx];

            let predict = forecast_md_m(
                data::VALUES[i],
                prev.l,
                prev.b,
                s_value,
                param_a,
                param_b,
                param_y,
                FI_DAMP,
                1.0,
            );

            curr_season_s_buffer[seasonal_idx] = predict.s;
            target[i] = predict.y;
            prev = predict;
        }

        // remember last 12 s values
        for i in 0..data::SEASON {
            prev_season_s[i] = curr_season_s_buffer[i];
        }

        return prev;
    }

    fn predict_test_next_values(
        test: &[f32],
        predicted: &mut [f32],
        prev_season_s: &[f32],
        prev: ForecastResult,
        param_a: f32,
        param_b: f32,
        param_y: f32,
    ) -> ForecastEval {
        let mut h = 1.0;

        for i in 0..predicted.len() {
            let predict = forecast_md_m(
                prev.y,
                prev.l,
                prev.b,
                prev_season_s[i],
                param_a,
                param_b,
                param_y,
                FI_DAMP,
                h,
            );
            predicted[i] = predict.y;
            h += 1.0;
        }
        return ForecastEval {
            mae: data::mae(predicted, test),
            rmse: data::rmse(data::mse(predicted, test)),
        };
    }

    fn predict_next_values(
        predicted: &mut [f32],
        prev_season_s: &[f32],
        prev: ForecastResult,
        param_a: f32,
        param_b: f32,
        param_y: f32,
    ) {
        let mut h = 1.0;
        let prev_s: [f32; data::SEASON] = [1.0; data::SEASON];

        for i in 0..predicted.len() {
            let predict = forecast_md_m(
                prev.y,
                prev.l,
                prev.b,
                prev_season_s[i],
                param_a,
                param_b,
                param_y,
                FI_DAMP,
                h,
            );
            predicted[i] = predict.y;
            h += 1.0;
        }
    }
}
