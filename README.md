# Rust Semantic Search with CLIP, Onnx and Qdrant

This project implements semantic search using the CLIP model. It's a web service built with Rust and Actix-web.

Clip model is run as text only mode. It is not using any image data.

Qdrant is used as a vector search engine. It is used to store the vectors and perform the search.

## Features

- Semantic search: Search for similar items based on semantic meaning.
- Health check: Check the status of the service.

## Getting Started

### Prerequisites

- Rust 1.74.1 or later
- Docker (optional)

### Environment Variables

- `QDRANT_HOST`: The host of the Qdrant instance.
- `QDRANT_API_KEY`: The API key of the Qdrant instance.
- `QDRANT_COLLECTION`: The name of the Qdrant collection. Defaults to `fashion`.
- `PORT`: The port of the web application. Defaults to `8080`

### Installation

1. Clone the repository:

```sh
git clone https://github.com/ugokalp/rust-semantic.git
cd rust-semantic
```

### Usage
1. Run the service:

```sh
cargo run
```
The service will start on localhost:8080.

### Build for Release:

```sh
cargo run --release
```

### Endpoints

`GET /`: The root endpoint. Returns a `200` status code and information about the service.

`GET /search`: Perform a semantic search. Accepts a `GET` request with a q query parameter for the search query and a `limit` query parameter for the number of results.

`GET /healthcheck`: Check the status of the service. Returns a `200` status code and a message if qdrant service is running correctly.
Otherwise returns a `500` status code and a message.


#### Example Responses

`GET /`:

```json
{
  "name": "rust-semantic",
  "commit_id": "72b5a62",
  "version": "0.1.0"
}
```

`GET /search?q=red%20shirt&limit=52`:

```json
{
  [
    {
        "id": "22138",
        "gender": "men"
    },
    {
        "id": "39942",
        "gender": "men"
    }
]
}
```

`GET /healthcheck`:

```json
{
  "status": "ok",
  "message": "All systems functional"
}
```
### Docker(Soon)
You can also run the service in a Docker container. Build the Docker image with:

```sh
docker build -t rust-semantic .
```

And run it with:
```sh
docker run -p 8080:8080 rust-semantic
```

### Contributing
Contributions are welcome!