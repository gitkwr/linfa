use linfa::{
    traits::{Fit, Predict},
    DatasetBase,
};
use linfa_logistic::LogisticRegression;
use ndarray::array;

fn main() {
    let x = array![[1.0], [0.0], [1.0], [0.0], [0.0]];
    let y = array![1, 1, 1, 1, 0];
    let dataset = DatasetBase::new(x, y);
    let model = LogisticRegression::default().fit(&dataset).unwrap();

    println!("model: {:?}", model);
    println!();
    let pred = model.predict(&dataset.records);
    println!("pred: {:?}", pred);
}
