use crate::render;
use std::ops::Range;

const SEASON: usize = 12;
const TRAIN_RANGE: Range<usize> = 0..SEASON * 4;
const TEST_RANGE: Range<usize> = SEASON * 4..SEASON * 5;

pub fn analyze() {
    let data: Vec<u32> = vec![
        953, 1018, 1295, 1448, 1584, 1792, 1837, 1608, 1491, 1357, 1381, 1078, 1138, 1095, 1375,
        1630, 1837, 1865, 2085, 1919, 1674, 1609, 1721, 1250, 1180, 1232, 1544, 1787, 1959, 2053,
        2250, 1932, 1775, 1743, 1688, 1246, 1341, 1301, 1677, 1894, 2065, 2098, 2466, 2125, 1936,
        1803, 1837, 1390, 1356, 1347, 1765, 1986, 2096, 2191, 2555, 2253, 2071, 1885, 2024, 1470,
    ];
    assert_eq!(data.len(), SEASON * 5, "data expected to have 60 elements");

    error_analysis(&data, "naive_mean", predict_mean);
    error_analysis(&data, "naive_last", predict_last);
    error_analysis(&data, "naive_last_season", predict_seasonal);
    render_graphs(data);
}

fn error_analysis(data: &[u32], method_name: &str, predict_fn: fn(&[u32]) -> u32) {
    let test = &data[TEST_RANGE];
    let mut train = data[TRAIN_RANGE].to_vec();
    let mut errors = Vec::new();

    for idx in 0..TEST_RANGE.len() {
        let predict = predict_fn(&train);
        let abs_diff = test[idx].max(predict) - test[idx].min(predict);
        train.push(test[idx]);
        errors.push(abs_diff);
    }

    let mae_mag = errors.iter().sum::<u32>();
    let mae = mae_mag / TEST_RANGE.len() as u32;

    let rmse_mag = errors.iter().map(|e| e * e).sum::<u32>() / TEST_RANGE.len() as u32;
    let rmse = f64::sqrt(rmse_mag as f64);

    println!("\nError analysis for `{}` method", method_name);
    println!("MAE: {mae}");
    println!("RMSE: {rmse}");
}

fn predict_mean(data: &[u32]) -> u32 {
    data.iter().sum::<u32>() / data.len() as u32
}

fn predict_last(data: &[u32]) -> u32 {
    data.last().copied().expect("non empty data")
}

fn predict_seasonal(data: &[u32]) -> u32 {
    let prev_season = data[data.len() - SEASON];
    prev_season
}

fn render_graphs(data: Vec<u32>) {
    println!("");
    // 0) visualize data
    render::render(&data, false, None, "starting_data");
    // 1) visualize seasons
    render::render(&data, true, None, "seasonal_dynamics");

    // lab_1 naive mean
    {
        let predict = predict_mean(&data);
        let mut forecast = data.clone();
        forecast.extend(std::iter::repeat(predict).take(SEASON * 3));
        render::render(&forecast, true, Some("lab_1"), "naive_mean");
    }

    // lab_1 naive last
    {
        let predict = predict_last(&data);
        let mut forecast = data.clone();
        forecast.extend(std::iter::repeat(predict).take(SEASON * 3));
        render::render(&forecast, true, Some("lab_1"), "naive_last");
    }

    // lab_1 naive last season
    {
        let mut forecast = data.clone();
        let last_season = &data[data.len() - SEASON..data.len()];
        for _ in 0..3 {
            forecast.extend_from_slice(last_season);
        }
        render::render(&forecast, true, Some("lab_1"), "naive_last_season");
    }
}
