use std::io::Error;

use ndarray::{Array1, ArrayBase, CowArray, Dim, OwnedRepr};
use ort::{
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtError, Session, SessionBuilder,
    Value,
};
use qdrant_client::client::{QdrantClient, QdrantClientConfig};
use tokenizers::tokenizer::Tokenizer;

pub fn load_tokenizer() -> tokenizers::tokenizer::Result<Tokenizer> {
    return Tokenizer::from_file("models/tokenizer.json");
}

pub fn load_text_model() -> Result<Session, OrtError> {
    let environment = Environment::builder()
        .with_name("CLIP-Textual")
        .with_execution_providers([ExecutionProvider::CPU(Default::default())])
        .with_log_level(ort::LoggingLevel::Fatal)
        .build()?
        .into_arc();

    let session = SessionBuilder::new(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_model_from_file("models/model.onnx");
    return session;
}

pub fn tokenize(
    tokenizer: Tokenizer,
    q: String,
) -> Result<ArrayBase<OwnedRepr<i64>, Dim<[usize; 1]>>, Error> {
    let encoding = tokenizer.encode(q, false).unwrap();
    let input_ids = encoding
        .get_ids()
        .iter()
        .map(|i| *i as i64)
        .collect::<Vec<_>>();
    let tokens = Array1::from_iter(input_ids.iter().cloned());
    return Ok(tokens);
}

pub fn run_text_model(
    session: &Session,
    tokenizer: &Tokenizer,
    q: String,
) -> Result<Vec<Value<'static>>, OrtError> {
    let tokenized = tokenize(tokenizer.clone(), q.clone()).unwrap();
    let tokens = CowArray::from(tokenized);
    let n_tokens = tokens.shape()[0];
    let array = tokens.clone().into_shape((1, n_tokens)).unwrap().into_dyn();
    let inputs = vec![Value::from_array(session.allocator(), &array)?];
    let outputs: Vec<Value> = session.run(inputs)?;

    Ok(outputs)
}

pub fn load_qdrant_client() -> QdrantClient {
    QdrantClientConfig::from_url(std::env::var("QDRANT_URL").unwrap().as_str())
        .with_api_key(std::env::var("QDRANT_API_KEY").unwrap().as_str())
        .build()
        .unwrap()
}
