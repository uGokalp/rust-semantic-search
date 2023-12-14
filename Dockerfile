# ---------------------------------------------------# 1 - Build Stage
#
# Use official rust image to for application build
# ---------------------------------------------------
FROM rust:1.74.1-slim as build

# Setup working directory
WORKDIR /usr/src/rust-semantic
COPY . .

# Install dependency
RUN apt-get update && apt-get install -y \
   g++


# Build application
RUN cargo install --path .

# ---------------------------------------------------
# 2 - Deploy Stage
#
# Use a distroless image for minimal container size
# - Copy application files into the image
# ---------------------------------------------------
FROM gcr.io/distroless/cc-debian11

# Set the architecture argument (arm64, i.e. aarch64 as default)
# For amd64, i.e. x86_64, you can append a flag when invoking the build `... --build-arg "ARCH=x86_64"`

# Application files
COPY --from=build /usr/local/cargo/bin/rust-semantic /usr/local/bin/rust-semantic
COPY --from=build /usr/local/cargo/bin/libonnxruntime.* /usr/local/bin


CMD ["rust-semantic"]