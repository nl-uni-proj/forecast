use crate::data;
use crate::render;

pub fn analyze() {
    error_analysis(data::VALUES, "naive_mean", predict_mean);
    error_analysis(data::VALUES, "naive_last", predict_last);
    error_analysis(data::VALUES, "naive_last_season", predict_seasonal);
    render_graphs(data::VALUES);
}

fn error_analysis(data: &[f32], method_name: &str, predict_fn: fn(&[f32]) -> f32) {
    let mut train = data[data::TRAIN_RANGE].to_vec();
    let test = &data[data::TEST_RANGE];
    let mut predicted = Vec::new();

    for idx in 0..data::TEST_RANGE.len() {
        let predict = predict_fn(&train);
        predicted.push(predict);
        train.push(test[idx]);
    }

    let mae = data::mae(&predicted, &test);
    let rmse = data::rmse(data::mse(&predicted, &test));

    println!("\nError analysis for `{}` method", method_name);
    println!("MAE: {mae}");
    println!("RMSE: {rmse}");
}

fn predict_mean(data: &[f32]) -> f32 {
    data.iter().sum::<f32>() / data.len() as f32
}

fn predict_last(data: &[f32]) -> f32 {
    data.last().copied().expect("non empty data")
}

fn predict_seasonal(data: &[f32]) -> f32 {
    let prev_season = data[data.len() - data::SEASON];
    prev_season
}

fn render_graphs(data: &[f32]) {
    println!("");
    // 0) visualize data
    render::render(&data, false, None, "starting_data");
    // 1) visualize seasons
    render::render(&data, true, None, "seasonal_dynamics");

    // naive mean
    {
        let predict = predict_mean(data);
        let mut forecast = data.to_vec();
        forecast.extend(std::iter::repeat(predict).take(data::SEASON * 3));
        render::render(&forecast, true, Some("lab_1"), "naive_mean");
    }

    // naive last
    {
        let predict = predict_last(data);
        let mut forecast = data.to_vec();
        forecast.extend(std::iter::repeat(predict).take(data::SEASON * 3));
        render::render(&forecast, true, Some("lab_1"), "naive_last");
    }

    // naive last season
    {
        let mut forecast = data.to_vec();
        let last_season = &data[data.len() - data::SEASON..data.len()];
        for _ in 0..3 {
            forecast.extend_from_slice(last_season);
        }
        render::render(&forecast, true, Some("lab_1"), "naive_last_season");
    }
}
