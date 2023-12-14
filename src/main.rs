mod ml;

use std::{sync::Mutex, time::Instant};

use actix_web::{
    get,
    middleware::Logger,
    web::{self, Data},
    App, HttpResponse, HttpServer, Responder,
};
use dotenv::dotenv;
use qdrant_client::{client::QdrantClient, qdrant::SearchPoints};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::ml::run_text_model;

struct AppState {
    app_name: String,
    tokenizer: Tokenizer,
    text_model: ort::Session,
    client: QdrantClient,
    commit_id: String,
}

#[derive(Deserialize)]
struct SearchQuery {
    q: String,
    limit: u64,
}

#[derive(Deserialize, Serialize, Debug)]
struct SearchResult {
    id: String,
    gender: String,
}
#[derive(Deserialize, Serialize)]
struct HealthCheck {
    status: String,
    msg: String,
}

#[get("/search")]
async fn search(data: Data<Mutex<AppState>>, item: web::Query<SearchQuery>) -> impl Responder {
    let state: std::sync::MutexGuard<'_, AppState> = data.lock().unwrap();

    let start_of_exec = Instant::now();
    let output = run_text_model(&state.text_model, &state.tokenizer, item.q.clone()).unwrap();
    let end_of_exec = Instant::now();
    let onnx_duration = end_of_exec.duration_since(start_of_exec);

    let collection_name = std::env::var("QDRANT_COLLECTION_NAME").unwrap_or("fashion".into());
    let vector = output[0].try_extract().unwrap();

    let start_of_exec = Instant::now();
    let res = state
        .client
        .search_points(&SearchPoints {
            collection_name,
            vector: vector.view().as_slice().unwrap().to_vec(),
            limit: item.limit.clone(),
            with_payload: Some(true.into()),
            ..Default::default()
        })
        .await
        .unwrap();
    let end_of_exec = Instant::now();
    let qdrant_duration = end_of_exec.duration_since(start_of_exec);

    let found_point: Vec<SearchResult> = res
        .result
        .iter()
        .map(|f| SearchResult {
            id: f.payload.get("id").unwrap().as_str().unwrap().to_string(),
            gender: f
                .payload
                .get("gender")
                .unwrap()
                .as_str()
                .unwrap()
                .to_string(),
        })
        .collect();
    HttpResponse::Ok()
        .append_header(("X-ONNX-Time", format!("{} ms", onnx_duration.as_millis())))
        .append_header((
            "X-Qdrant-Time",
            format!("{} ms", qdrant_duration.as_millis()),
        ))
        .json(found_point)
}

#[get("/")]
async fn index(data: web::Data<Mutex<AppState>>) -> impl Responder {
    let app_name = data.lock().unwrap().app_name.clone();
    let commit_id = data.lock().unwrap().commit_id.clone();
    let resp: Value = serde_json::from_str(
        format!(
            r#"{{
            "app_name": "{app_name}",
            "version": "0.1.0",
            "commit_id": "{commit_id}"
            }}"#,
            app_name = app_name,
            commit_id = commit_id
        )
        .as_str(),
    )
    .expect("Failed to parse JSON");
    HttpResponse::Ok().json(resp)
}

#[get("/healthcheck")]
async fn healthcheck(data: web::Data<Mutex<AppState>>) -> impl Responder {
    let qdrant_health = match data.lock().unwrap().client.health_check().await {
        Ok(_) => HttpResponse::Ok().json(HealthCheck {
            status: "ok".into(),
            msg: "All systems functional".into(),
        }),
        Err(_) => HttpResponse::Ok().json(HealthCheck {
            status: "ok".into(),
            msg: "Qdrant is not available".into(),
        }),
    };
    qdrant_health
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    dotenv().ok();
    let port: u16 = std::env::var("PORT")
        .unwrap_or(3000.to_string())
        .parse::<u16>()
        .unwrap();

    env_logger::init();
    log::info!("Starting server, Running on port: {}", port);

    let commit_id = std::process::Command::new("git")
        .args(&["rev-parse", "--short", "HEAD"])
        .output()
        .expect("failed to execute git command")
        .stdout;

    HttpServer::new(move || {
        let app_data = Data::new(Mutex::new(AppState {
            app_name: "rust-semantic".into(),
            tokenizer: ml::load_tokenizer().unwrap(),
            text_model: ml::load_text_model().unwrap(),
            client: ml::load_qdrant_client(),
            commit_id: String::from_utf8(commit_id.clone()).unwrap().trim().into(),
        }));

        App::new()
            .wrap(Logger::default())
            .app_data(Data::clone(&app_data))
            .service(search)
            .service(index)
            .service(healthcheck)
    })
    .workers(1)
    .bind(("127.0.0.1", port))?
    .run()
    .await
}
