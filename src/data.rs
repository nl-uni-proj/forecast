use std::ops::Range;

pub const SEASON: usize = 12;
pub const TRAIN_RANGE: Range<usize> = 0..SEASON * 4;
pub const TEST_RANGE: Range<usize> = SEASON * 4..SEASON * 5;

pub const VALUES: &[f32] = &[
    953.0, 1018.0, 1295.0, 1448.0, 1584.0, 1792.0, 1837.0, 1608.0, 1491.0, 1357.0, 1381.0, 1078.0,
    1138.0, 1095.0, 1375.0, 1630.0, 1837.0, 1865.0, 2085.0, 1919.0, 1674.0, 1609.0, 1721.0, 1250.0,
    1180.0, 1232.0, 1544.0, 1787.0, 1959.0, 2053.0, 2250.0, 1932.0, 1775.0, 1743.0, 1688.0, 1246.0,
    1341.0, 1301.0, 1677.0, 1894.0, 2065.0, 2098.0, 2466.0, 2125.0, 1936.0, 1803.0, 1837.0, 1390.0,
    1356.0, 1347.0, 1765.0, 1986.0, 2096.0, 2191.0, 2555.0, 2253.0, 2071.0, 1885.0, 2024.0, 1470.0,
];

// mean square error
pub fn mse(predicted: &[f32], test: &[f32]) -> f32 {
    assert_eq!(predicted.len(), test.len());

    let mut error_sum = 0.0;
    for i in 0..predicted.len() {
        let diff = predicted[i] - test[i];
        error_sum += diff * diff;
    }
    error_sum / predicted.len() as f32
}

// mean average error
pub fn mae(predicted: &[f32], test: &[f32]) -> f32 {
    assert_eq!(predicted.len(), test.len());

    let mut error_sum = 0.0;
    for i in 0..predicted.len() {
        let diff = predicted[i] - test[i];
        error_sum += diff;
    }
    (error_sum / predicted.len() as f32).abs()
}

// root mean square error
pub fn rmse(mse: f32) -> f32 {
    mse.sqrt()
}

pub fn trend_average(data: &[f32]) -> f32 {
    let mut trend_sum: f32 = 0.0;
    for i in 1..data.len() {
        let trend = data[i] - data[i - 1];
        trend_sum += trend;
    }
    trend_sum / (data.len() - 1) as f32
}
